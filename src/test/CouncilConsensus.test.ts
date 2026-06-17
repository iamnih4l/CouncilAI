import { describe, it, expect } from 'vitest';
import { runCouncilConsensus } from '../services/inference/CouncilConsensus';
import { PathologyClass } from '../services/inference/types';
import type { DenseNetOutput, AttentionNetOutput, SwinUNETROutput } from '../services/inference/types';

describe('CouncilConsensus', () => {
  it('should compute agreement and final severity correctly when models agree', () => {
    const densenet = {
      modelName: 'DenseNet-121',
      confidence: 90,
      primaryDiagnosis: PathologyClass.GLIOBLASTOMA,
      inferenceTimeMs: 100
    } as DenseNetOutput;
    
    const attention = {
      modelName: 'Attention-Net',
      confidence: 88,
      focusCenter: [50, 50, 50],
      focusRadius: 10,
      focusRegion: 'FRONTAL_LEFT',
      inferenceTimeMs: 120
    } as AttentionNetOutput;
    
    const swin = {
      modelName: 'Swin-UNETR',
      confidence: 92,
      tumorCenter: [51, 49, 50],
      tumorRadius: 12,
      tumorVolumeMm3: 500,
      segmentedRegions: [],
      inferenceTimeMs: 150
    } as unknown as SwinUNETROutput;

    const consensus = runCouncilConsensus(densenet, attention, swin, 'MRI');
    
    expect(consensus.primaryDiagnosis).toBe(PathologyClass.GLIOBLASTOMA);
    expect(consensus.severity).toBe('critical'); // High confidence tumor -> critical
    expect(consensus.modelAgreement).toBeGreaterThan(0.8);
    expect(consensus.overallConfidence).toBeGreaterThan(85);
    expect(consensus.councilNotes.length).toBeGreaterThan(0);
  });

  it('should flag low agreement when spatial coordinates are far apart', () => {
    const densenet = {
      modelName: 'DenseNet-121',
      confidence: 80,
      primaryDiagnosis: PathologyClass.NORMAL,
      inferenceTimeMs: 100
    } as DenseNetOutput;
    
    const attention = {
      modelName: 'Attention-Net',
      confidence: 70,
      focusCenter: [10, 10, 10],
      focusRadius: 5,
      focusRegion: 'FRONTAL_LEFT',
      inferenceTimeMs: 120
    } as AttentionNetOutput;
    
    const swin = {
      modelName: 'Swin-UNETR',
      confidence: 85,
      tumorCenter: [90, 90, 90], // Far apart
      tumorRadius: 5,
      tumorVolumeMm3: 10,
      segmentedRegions: [],
      inferenceTimeMs: 150
    } as unknown as SwinUNETROutput;

    const consensus = runCouncilConsensus(densenet, attention, swin, 'CT');
    
    // Spatial disagreement should lower agreement score
    expect(consensus.modelAgreement).toBeLessThan(0.7);
    expect(consensus.councilNotes.some(note => note.includes('divergent') || note.includes('LOW AGREEMENT'))).toBe(true);
  });
});
