import { useState, useEffect, useCallback } from 'react';
import type { CouncilConsensusResult } from '../services/inference/types';
import { StorageService } from '../utils/storage';

export interface PersistentReport {
  id: string;
  patientName?: string;
  modality: 'MRI' | 'CT' | 'X-Ray';
  date: string;
  result: CouncilConsensusResult;
}

export function usePersistentReports() {
  const [reports, setReports] = useState<PersistentReport[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Load reports on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const stored = await StorageService.getReports();
        setReports(stored);
      } catch (err) {
        console.error('Failed to parse diagnostic vault:', err);
      } finally {
        setIsLoading(false);
      }
    };
    loadData();
  }, []);

  const addReport = useCallback(async (result: CouncilConsensusResult) => {
    const newReport: PersistentReport = {
      id: `PT-${Math.floor(Math.random() * 9000) + 1000}`,
      modality: 'MRI', // Default for now, can be parameterized
      date: new Date().toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      }),
      result,
    };

    await StorageService.saveReport(newReport);
    setReports((prev) => [newReport, ...prev].slice(0, 100));
    
    return newReport;
  }, []);

  const deleteReport = useCallback(async (id: string) => {
    await StorageService.deleteReport(id);
    setReports((prev) => prev.filter(r => r.id !== id));
  }, []);

  const clearAll = useCallback(async () => {
    await StorageService.clearReports();
    setReports([]);
  }, []);

  return { reports, isLoading, addReport, deleteReport, clearAll };
}
