use rand::prelude::*;

pub trait Optimizer {
    type OptimizerInstance: OptimizerInstance;

    fn instance(self, n_vars: usize) -> Self::OptimizerInstance;
}

pub trait OptimizerInstance {
    fn apply<'a>(&mut self, vars_and_grads: impl Iterator<Item = (&'a mut f32, &'a f32)>);
}

pub trait StatelessOptimizer: OptimizerInstance {}

#[derive(Debug)]
pub struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Layer {
        let mut source = rand::thread_rng();
        let mut weights = Vec::with_capacity(output_size * input_size);

        for _ in 0..output_size {
            for _ in 0..input_size {
                weights.push((source.gen::<u64>() % 100) as f32 / 1000.0)
            }
        }

        let mut biases = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            biases.push((source.gen::<u64>() % 100) as f32 / 100000.0)
        }

        Layer {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    pub fn run(&self, input: &[f32], output: &mut [f32]) {
        assert!(input.len() == self.input_size);
        assert!(output.len() == self.output_size);

        for (i, activation) in output.iter_mut().enumerate() {
            // the dot product of weights and inputs
            let net_input = input
                .iter()
                .enumerate()
                .map(|(j, input)| self.weights[i * self.input_size + j] * input)
                .sum::<f32>();
            let biased_net_input = net_input + self.biases[i];
            *activation = biased_net_input.max(0.0);
        }
    }
}

pub struct Model {
    layers: Vec<Layer>,
    outputs: Vec<Vec<f32>>,
    input_size: usize,
    output_size: usize,
}

impl Model {
    pub fn new(dimensions: &[usize]) -> Model {
        assert!(dimensions.len() > 2);
        assert!(!dimensions.contains(&0));

        let mut layers = Vec::with_capacity(dimensions.len() - 1);
        for i in 1..dimensions.len() {
            layers.push(Layer::new(dimensions[i - 1], dimensions[i]));
        }

        Model {
            outputs: layers.iter().map(|l| vec![0.0; l.output_size]).collect(),
            layers,
            input_size: *dimensions.first().unwrap(),
            output_size: *dimensions.last().unwrap(),
        }
    }

    pub fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        assert!(input.len() == self.input_size);

        let mut ninput = input;
        for (layer, output) in self.layers.iter_mut().zip(self.outputs.iter_mut()) {
            layer.run(ninput, output);
            ninput = output;
        }

        ninput
    }

    pub fn train<'a>(
        &mut self,
        cases: impl Iterator<Item = &'a (impl AsRef<[f32]> + 'a, impl AsRef<[f32]> + 'a)>,
        optimizer: impl Optimizer,
        batch_size: usize,
    ) {
        let mut optimizer = optimizer.instance(
            self.layers
                .iter()
                .map(|l| l.output_size * l.input_size + l.output_size)
                .sum(),
        );

        // keep track of the gradient over each mini-batch, start it at 0
        let mut dw = self
            .layers
            .iter()
            .map(|l| vec![0.0; l.output_size * l.input_size])
            .collect::<Vec<_>>();
        let mut db = self
            .layers
            .iter()
            .map(|l| vec![0.0; l.output_size])
            .collect::<Vec<_>>();

        // $\delta_j$ from the Wikipedia article on backpropagation
        let mut delta = Vec::with_capacity(self.output_size);
        let mut delta_next = Vec::new();

        // keep track of the number of trials thus far
        let mut t = 0;

        for (input, target) in cases {
            let (input, target): (&[f32], &[f32]) = (input.as_ref(), target.as_ref());

            t += 1;

            self.infer(input);

            delta.clear();
            for (i, target) in target.iter().enumerate() {
                // derivative of 1/2 of squared error
                let dsqe = self.outputs.last().unwrap()[i] - target;
                delta.push(dsqe);
            }

            for (l, layer) in self.layers.iter().enumerate().rev() {
                assert!(delta.len() == layer.output_size);

                let prev_output = if l != 0 {
                    self.outputs.get(l - 1)
                } else {
                    None
                };

                // calculate gradient
                let dw = &mut dw[l];
                let db = &mut db[l];
                for i in 0..layer.output_size {
                    for j in 0..layer.input_size {
                        let activation = prev_output.map(|o| o[j]).unwrap_or(input[j]);

                        dw[i * layer.input_size + j] += delta[i] * activation;
                    }

                    db[i] += delta[i]
                }

                // calculate prev layer's error (it's the next one we'll be visiting)
                if let Some(prev_output) = prev_output {
                    delta_next.reserve(layer.input_size);
                    for (j, prev_output) in prev_output.iter().copied().enumerate() {
                        let total_err = if prev_output > 0.0 {
                            delta
                                .iter()
                                .enumerate()
                                .map(|(i, delta)| layer.weights[i * layer.input_size + j] * delta)
                                .sum::<f32>()
                        } else {
                            0.0
                        };

                        delta_next.push(total_err / delta.len() as f32)
                    }

                    delta.clear();
                    delta.append(&mut delta_next);
                }
            }

            if t % batch_size != 0 {
                continue;
            }

            // average it
            dw.iter_mut()
                .flat_map(|l| l.iter_mut())
                .for_each(|dw| *dw /= batch_size as f32);
            db.iter_mut()
                .flat_map(|l| l.iter_mut())
                .for_each(|db| *db /= batch_size as f32);

            let x_iter = self
                .layers
                .iter_mut()
                .flat_map(|l| l.weights.iter_mut().chain(l.biases.iter_mut()));
            let dx_iter = dw
                .iter()
                .zip(db.iter())
                .flat_map(|(dw, db)| dw.iter().chain(db.iter()));
            optimizer.apply(x_iter.zip(dx_iter));

            // reset weight/bias grads
            dw.iter_mut()
                .flat_map(|l| l.iter_mut())
                .for_each(|dw| *dw = 0.0);
            db.iter_mut()
                .flat_map(|l| l.iter_mut())
                .for_each(|db| *db = 0.0);
        }

        // don't re-optimize if we just did
        if t % batch_size == 0 {
            return;
        }

        // average it
        dw.iter_mut()
            .flat_map(|l| l.iter_mut())
            .for_each(|dw| *dw /= (t % batch_size) as f32);
        db.iter_mut()
            .flat_map(|l| l.iter_mut())
            .for_each(|db| *db /= (t % batch_size) as f32);

        let x_iter = self
            .layers
            .iter_mut()
            .flat_map(|l| l.weights.iter_mut().chain(l.biases.iter_mut()));
        let dx_iter = dw
            .iter()
            .zip(db.iter())
            .flat_map(|(dw, db)| dw.iter().chain(db.iter()));
        optimizer.apply(x_iter.zip(dx_iter));
    }
}

impl<T: StatelessOptimizer> Optimizer for T {
    type OptimizerInstance = Self;

    fn instance(self, _n_vars: usize) -> Self {
        self
    }
}

pub struct GradientDescentOptimizer {
    a: f32,
}

impl GradientDescentOptimizer {
    pub fn new(learn_rate: f32) -> GradientDescentOptimizer {
        GradientDescentOptimizer { a: learn_rate }
    }
}

impl StatelessOptimizer for GradientDescentOptimizer {}

impl OptimizerInstance for GradientDescentOptimizer {
    fn apply<'a>(&mut self, vars_and_grads: impl Iterator<Item = (&'a mut f32, &'a f32)>) {
        for (x, dx) in vars_and_grads {
            *x -= self.a * dx
        }
    }
}

pub struct AdamOptimizer {
    pub a: f32,
    pub b1: f32,
    pub b2: f32,
}

impl Optimizer for AdamOptimizer {
    type OptimizerInstance = AdamOptimizerInstance;

    fn instance(self, n_vars: usize) -> Self::OptimizerInstance {
        AdamOptimizerInstance {
            cfg: self,
            t: 0,
            vdx: vec![0.0; n_vars],
            sdx: vec![0.0; n_vars],
        }
    }
}

pub struct AdamOptimizerInstance {
    cfg: AdamOptimizer,
    t: i32,
    vdx: Vec<f32>,
    sdx: Vec<f32>,
}

impl OptimizerInstance for AdamOptimizerInstance {
    fn apply<'a>(&mut self, vars_and_grads: impl Iterator<Item = (&'a mut f32, &'a f32)>) {
        self.t += 1;

        for (((x, dx), vdx), sdx) in vars_and_grads
            .zip(self.vdx.iter_mut())
            .zip(self.sdx.iter_mut())
        {
            *vdx = self.cfg.b1 * *vdx + (1.0 - self.cfg.b1) * dx;
            *sdx = self.cfg.b2 * *sdx + (1.0 - self.cfg.b2) * dx * dx;

            let vdx_corr = *vdx / (1.0 - self.cfg.b1.powi(self.t));
            let sdx_corr = *sdx / (1.0 - self.cfg.b2.powi(self.t));

            *x -= self.cfg.a * vdx_corr / (sdx_corr.sqrt() + 1e-8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_sin() {
        let mut x = 0.0f32;
        let mut cases = Vec::new();
        while x < 1.0 {
            cases.push(([x], [(x * std::f32::consts::TAU).sin() + 1.0]));
            x += 0.0001;
        }

        let mut model = Model::new(&[1, 128, 1]);
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            cases.shuffle(&mut rng);

            model.train(
                cases.iter(),
                AdamOptimizer {
                    a: 0.01,
                    b1: 0.9,
                    b2: 0.999,
                },
                32,
            );
        }

        let mut x = 0.0;
        while x < 1.0 {
            let abs_diff = ((x * std::f32::consts::TAU).sin() - model.infer(&[x])[0] + 1.0).abs();
            assert!(abs_diff < 0.1, "the approximation must be accurate");
            x += 0.01;
        }
    }
}
