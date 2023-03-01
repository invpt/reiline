use std::{
    collections::{hash_map::Entry, HashMap},
    fs::File,
    io::{BufWriter, Write},
};

use rand::prelude::*;

use crate::bitset::BitSet;

mod bitset;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct State<'a> {
    places: BitSet<'a, usize>,
    pieces: BitSet<'a, usize>,
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct Action {
    piece: usize,
    pos: usize,
}

#[derive(Debug)]
struct ActionResult {
    q: f64,
}

fn reward(state: &State) -> f64 {
    if state.pieces.len() != 0 {
        return 0.0;
    }

    // find first nonempty square
    let mut i = None;
    for pos in state.places.mapping().iter().copied() {
        if !state.places.contains(&pos) {
            i = Some(pos);
            break;
        }
    }

    let Some(first_nonempty) = i else { return 0.0 };

    // find last nonempty square
    let mut last_nonempty = first_nonempty;
    for pos in state.places.mapping().iter().rev().copied() {
        if !state.pieces.contains(&pos) {
            last_nonempty = pos;
            break;
        }
    }

    let possible = state.pieces.mapping().iter().sum::<usize>();

    (state.places.mapping()[state.places.mapping().len() - 1] - state.places.mapping()[0]) as f64
        + possible as f64
        - last_nonempty as f64
        + first_nonempty as f64
        - 1.0
}

fn transition(state: &mut State, action: &Action) -> bool {
    if !state.pieces.contains(&action.piece) {
        return false;
    }

    for pos in action.pos..action.pos + action.piece {
        if !state.places.contains(&pos) {
            return false;
        }
    }

    for pos in action.pos..action.pos + action.piece {
        state.places.remove(&pos);
    }

    state.pieces.remove(&action.piece);

    true
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // parameters
    let places_mapping = (0..6).collect::<Vec<_>>(); // where can things be placed?
    let pieces_mapping = Vec::from_iter([1, 2, 3]); // what can be placed?
    let learn_rate = 0.8; // how fast will the table change based on new trials?
    let future_weight = 0.75; // how much will the future be weighted for each q-value?
    let trials = 10000; // how many random trials will be run?

    let mut qtab = HashMap::<State, HashMap<Action, ActionResult>>::new();
    let mut rng = rand::thread_rng();
    'outer: for tr in 0..trials {
        if tr % (trials / 100) == 0 {
            eprintln!("{}%", tr / (trials / 100));
        }

        let mut state = State {
            places: BitSet::all(&places_mapping),
            pieces: BitSet::all(&pieces_mapping),
        };

        while state.pieces.len() > 0 {
            let old_state = state.clone();
            let mut action;
            let mut cnt = 0;
            while {
                let piece = *state.pieces.iter().choose(&mut rng).unwrap();
                let pos = *state.places.iter().choose(&mut rng).unwrap();
                action = Action { piece, pos };
                cnt += 1;
                cnt != 100 && !transition(&mut state, &action)
            } {}

            if cnt == 100 {
                continue 'outer;
            }

            let rew = reward(&state);

            let future = qtab
                .get(&state)
                .unwrap_or(&HashMap::new())
                .values()
                .map(|r| r.q)
                .fold(f64::NEG_INFINITY, |a, b| a.max(b));

            let future = if future.is_infinite() { 0.0 } else { future };

            match qtab.entry(old_state) {
                Entry::Occupied(mut state_entry) => match state_entry.get_mut().entry(action) {
                    Entry::Occupied(mut q_entry) => {
                        q_entry.insert(ActionResult {
                            q: (1.0 - learn_rate) * q_entry.get().q
                                + learn_rate * (rew + future_weight * future),
                        });
                    }
                    Entry::Vacant(q_entry) => {
                        q_entry.insert(ActionResult {
                            q: learn_rate * (rew + future_weight * future),
                        });
                    }
                },
                Entry::Vacant(state_entry) => {
                    state_entry.insert(HashMap::from_iter([(
                        action,
                        ActionResult {
                            q: learn_rate * (rew + future_weight * future),
                        },
                    )]));
                }
            }
        }
    }

    {
        let mut csv = BufWriter::new(File::create("out.csv")?);

        csv.write_all(b"places,pieces,piece,pos,q\n")?;
        for (state, actions) in &qtab {
            for (action, ActionResult { q }) in actions {
                csv.write_all(
                    format!(
                        "{:?},{:?},{},{},{}\n",
                        state.places, state.pieces, action.piece, action.pos, q
                    )
                    .as_bytes(),
                )?;
            }
        }
    }

    Ok(())
}
