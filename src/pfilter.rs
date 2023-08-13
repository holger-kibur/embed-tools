extern crate alloc;

use alloc::vec::Vec;
use nalgebra::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

type FiltFloat = f32;
type StateVector = DVector<FiltFloat>;
type ObserveVector = DVector<FiltFloat>;
type PropagateFunc = fn(StateVector) -> StateVector;
type ObserveFunc = fn(StateVector) -> ObserveVector;

static NUM_PARTICLES: usize = 25;

enum FrameAction {
    Propagate,
    Resample,
}

struct ParticleFilterFrame {
    particles: DMatrix<FiltFloat>,
    action: FrameAction,
    covar: DMatrix<FiltFloat>,
}

pub struct ParticleFilter {
    frames: Vec<ParticleFilterFrame>,
    state_dim: usize,
    observe_dim: usize,
    prop_func: PropagateFunc,
    obsv_func: ObserveFunc,
    rng: SmallRng,
}

impl ParticleFilter {
    pub fn new(
        state_dim: usize,
        observe_dim: usize,
        prop_func: PropagateFunc,
        obsv_func: ObserveFunc,
    ) -> Self {
        ParticleFilter {
            frames: Vec::new(),
            state_dim,
            observe_dim,
            prop_func,
            obsv_func,
            rng: SmallRng::seed_from_u64(10),
        }
    }

    pub fn init(&mut self, mean: StateVector, std: FiltFloat) {
        assert_eq!(mean.len(), self.state_dim);
        let mut particles = DMatrix::from_row_iterator(
            NUM_PARTICLES,
            self.state_dim,
            (&mut self.rng).sample_iter::<FiltFloat, StandardNormal>(StandardNormal),
        );
        particles.row_iter_mut().for_each(|mut row| row += mean.transpose());
        self.frames.push(ParticleFilterFrame {
            particles: particles,
            action: FrameAction::Resample,
            covar: DMatrix::identity(self.state_dim, self.state_dim) * std,
        });
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use nalgebra::{DVector, DMatrix, Matrix};
    use serde::{Serialize, Deserialize};
    use core::fmt::Display;
    use std::{*, io::Write};

    use crate::pfilter::NUM_PARTICLES;

    use super::{ParticleFilter, ParticleFilterFrame, FiltFloat};

    fn dump_matrix(stream: &mut dyn Write, matrix: &DMatrix<FiltFloat>, label: &str) -> Result<(), io::Error>{
        writeln!(stream, "+{:-^24}+", label)?;
        for i in 0..matrix.nrows() {
            write!(stream, "|  ")?;
            for j in 0..matrix.ncols() {
                write!(stream, "{:^9.3}  ", matrix[(i, j)])?;
            }
            writeln!(stream, "|")?;
        }
        writeln!(stream, "+{:-^24}+", "")?;
        Ok(())
    }

    fn dump_filter_state(filter: &ParticleFilter) {
        let mut dumpfile = fs::File::create("dumpfile.txt").unwrap();
        for frame in filter.frames.iter() {
            dump_matrix(&mut dumpfile, &frame.covar, "Covar").unwrap();
            dump_matrix(&mut dumpfile, &frame.particles, "Particles").unwrap();
        }
    }

    #[test]
    fn init_sample() -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        let mut filter = ParticleFilter::new(2, 1, |_| DVector::zeros(2), |_| DVector::zeros(1));
        filter.init(DVector::from_row_slice(&[0.0, 0.0]), 1.0);
        dump_filter_state(&filter);
        Ok(())
    }
}
