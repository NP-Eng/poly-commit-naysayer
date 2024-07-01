#![cfg_attr(not(feature = "std"), no_std)]

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly::Polynomial;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};

use ark_std::{fmt::Debug, rand::RngCore};

pub mod linear_codes;

mod utils;

#[cfg(test)]
mod tests;

type NaysayerError = ark_poly_commit::Error;

/// Naysayer interface for polynomial commitment schemes
pub trait PCSNaysayer<F, P>: PolynomialCommitment<F, P>
where
    F: PrimeField,
    P: Polynomial<F>,
{
    type NaysayerProof: PartialEq + Debug;

    /// Naysays the given proof, returning
    /// - Err(e) if controlled execution error e was encountered
    /// - Ok(None) if no errors were encountered and the proof is valid ("aye")
    /// - Ok(Some(naysayer_proof)) if an assertion error was encountered
    fn naysay<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = F>,
        proof: &Self::Proof,
        sponge: &mut impl CryptographicSponge,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<Option<Self::NaysayerProof>, NaysayerError>
    where
        Self::Commitment: 'a;

    /// Verifies the naysayer proof. Returns:
    /// - Ok(true) if the original proof is rejected (i.e. the naysayer proof
    ///   points to a valid issue).
    /// - Ok(false) if the original proof is not rejected, i.e. the naysayer
    ///   proof points to a non-issue
    /// - Err if another type of error occurs during verification of the
    ///   naysayer proof.
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
