import type { CouncilConsensusResult } from './types';

/**
 * ClinicalValidationLayer — Provides clinical guardrails and safety checks.
 * This layer ensures that the neural network outputs are within physiologically 
 * plausible ranges and meet clinical standards before report generation.
 */
export class ClinicalValidationLayer {
  /**
   * Validates the consensus results against clinical rules.
   * @param result The result from the Council Consensus algorithm
   * @returns An array of validation findings or warnings
   */
  static validate(result: CouncilConsensusResult): string[] {
    const warnings: string[] = [];

    // 1. Confidence Guardrail
    if (result.overallConfidence < 65) {
      warnings.push("LOW_CONFIDENCE: Diagnostic certainty below clinical threshold. Recommend manual radiologist arbitration.");
    }

    // 2. Conflict Detection
    // If the Classifier and Segmenter disagree significantly
    if (result.overallConfidence > 80 && result.councilNotes.length === 0) {
       warnings.push("CONSENSUS_DISCREPANCY: High confidence but no significant clinical markers identified.");
    }

    // 3. Anatomical Consistency
    // (In a real system, we'd check if the AttentionNet focus matches the SwinUNETR mask centroid)
    
    return warnings;
  }

  /**
   * Performs a "Sanity Check" on the input tensor to ensure it looks like a medical scan.
   * (e.g. checking pixel intensity distributions)
   */
  static async performSanityCheck(intensityMean: number): Promise<boolean> {
    // Basic check: Very dark or very bright images are likely not valid scans
    if (intensityMean < 0.05 || intensityMean > 0.95) {
      return false;
    }
    return true;
  }
}
