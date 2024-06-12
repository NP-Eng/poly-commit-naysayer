use ark_std::rand::RngCore;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly::Polynomial;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};

mod linear_codes;
mod utils;

type NaysayerError = ark_poly_commit::Error;

pub trait PCSNaysayer<F, P>: PolynomialCommitment<F, P>
where
    F: PrimeField,
    P: Polynomial<F>,
{
    type NaysayerProof;

    fn naysay<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = F>,
        proof: &Self::Proof,
        sponge: &mut impl CryptographicSponge,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::NaysayerProof, NaysayerError>
    where
        Self::Commitment: 'a;

    fn verify_naysay<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = F>,
        proof_array: &Self::Proof,
        naysayer_proof: &Self::NaysayerProof,
        sponge: &mut impl CryptographicSponge,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, NaysayerError>
    where
        Self::Commitment: 'a;
}
