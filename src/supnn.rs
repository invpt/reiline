use rand::prelude::*;

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
            biases.push((source.gen::<u64>() % 100) as f32 / 1000000.0)
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

    pub fn adjust(&mut self, wg_adj: &[f32], b_adj: &[f32]) {
        assert!(wg_adj.len() == self.weights.len());

        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.weights[i * self.input_size + j] += wg_adj[i * self.input_size + j]
            }

            self.biases[i] += b_adj[i]
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
        learn_rate: f32,
        bias_learn_rate: f32,
    ) {
        let mut desired_changes: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());
        let mut bias_changes: Vec<Vec<f32>> = Vec::with_capacity(self.layers.len());
        for l in 0..self.layers.len() {
            desired_changes.push(vec![
                0.0;
                self.layers[l].output_size * self.layers[l].input_size
            ]);
            bias_changes.push(vec![0.0; self.layers[l].output_size]);
        }

        let mut case_count = 0;
        // $\delta_j$ from the Wikipedia article on backpropagation
        let mut delta = Vec::with_capacity(self.output_size);
        let mut delta_next = Vec::new();
        for (input, target) in cases {
            let (input, target): (&[f32], &[f32]) = (input.as_ref(), target.as_ref());
            case_count += 1;

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

                // weight and bias nudges
                let wch = &mut desired_changes[l];
                let bch = &mut bias_changes[l];
                for i in 0..layer.output_size {
                    for j in 0..layer.input_size {
                        let v = if let Some(prev_output) = prev_output {
                            prev_output[j]
                        } else {
                            input[j]
                        };

                        wch[i * layer.input_size + j] -= learn_rate * delta[i] * v;
                    }

                    bch[i] -= bias_learn_rate * delta[i]
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

            if case_count % 30 != 0 {
                continue;
            }

            // average it
            for layer_wg_ch in desired_changes.iter_mut() {
                for wg_ch in layer_wg_ch.iter_mut() {
                    *wg_ch /= case_count as f32;
                }
            }

            for layer_bias_ch in bias_changes.iter_mut() {
                for bias_ch in layer_bias_ch.iter_mut() {
                    *bias_ch /= case_count as f32;
                }
            }

            for ((layer, wch), bch) in self
                .layers
                .iter_mut()
                .zip(desired_changes.iter())
                .zip(bias_changes.iter())
            {
                layer.adjust(wch, bch)
            }

            case_count = 0;

            for layer_wg_ch in desired_changes.iter_mut() {
                for wg_ch in layer_wg_ch.iter_mut() {
                    *wg_ch = 0.0;
                }
            }

            for layer_bias_ch in bias_changes.iter_mut() {
                for bias_ch in layer_bias_ch.iter_mut() {
                    *bias_ch = 0.0;
                }
            }
        }

        // average it
        for layer_wg_ch in desired_changes.iter_mut() {
            for wg_ch in layer_wg_ch.iter_mut() {
                *wg_ch /= case_count as f32;
            }
        }

        for layer_bias_ch in bias_changes.iter_mut() {
            for bias_ch in layer_bias_ch.iter_mut() {
                *bias_ch /= case_count as f32;
            }
        }

        for ((layer, wch), bch) in self
            .layers
            .iter_mut()
            .zip(desired_changes.iter())
            .zip(bias_changes.iter())
        {
            layer.adjust(wch, bch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut x = 0.0f32;
        let mut cases = Vec::new();
        while x < 1.0 {
            cases.push(([x], [x * x]));
            x += 0.001;
        }

        let mut model = Model::new(&[1, 128, 1]);
        let mut rng = rand::thread_rng();
        loop {
            cases.shuffle(&mut rng);

            let mut x = 0.0;
            while x < 1.0 {
                println!(
                    "inp: {:?}\ninf: {:?}\nact: {:?}",
                    x,
                    model.infer(&[x])[0],
                    x * x
                );
                x += 0.1
            }

            model.train(cases.iter(), 0.05, 0.01);
        }
    }
}
