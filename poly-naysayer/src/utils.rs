use ark_ff::PrimeField;

#[cfg(feature = "parallel")]
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    prelude::IndexedParallelIterator,
};

#[inline]
pub(crate) fn inner_product<F: PrimeField>(v1: &[F], v2: &[F]) -> F {
    ark_std::cfg_iter!(v1)
        .zip(v2)
        .map(|(li, ri)| *li * ri)
        .sum()
}
