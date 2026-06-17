import localforage from 'localforage';
import type { PersistentReport } from '../hooks/usePersistentReports';

// Initialize the CouncilMed DB instance
const reportStore = localforage.createInstance({
  name: 'CouncilMed',
  storeName: 'reports_vault',
  description: 'Secure, offline-first storage for clinical reports'
});

const SETTINGS_STORE = localforage.createInstance({
  name: 'CouncilMed',
  storeName: 'workstation_settings'
});

export const StorageService = {
  // --- Reports ---
  async saveReport(report: PersistentReport): Promise<void> {
    try {
      const existing = await this.getReports();
      const updated = [report, ...existing].slice(0, 100); // Cap at 100
      await reportStore.setItem('reports', updated);
    } catch (error) {
      console.error('[StorageService] Error saving report:', error);
      throw error;
    }
  },

  async getReports(): Promise<PersistentReport[]> {
    try {
      const data = await reportStore.getItem<PersistentReport[]>('reports');
      return data || [];
    } catch (error) {
      console.error('[StorageService] Error fetching reports:', error);
      return [];
    }
  },

  async deleteReport(id: string): Promise<void> {
    try {
      const existing = await this.getReports();
      const updated = existing.filter(r => r.id !== id);
      await reportStore.setItem('reports', updated);
    } catch (error) {
      console.error('[StorageService] Error deleting report:', error);
      throw error;
    }
  },

  async clearReports(): Promise<void> {
    try {
      await reportStore.removeItem('reports');
    } catch (error) {
      console.error('[StorageService] Error clearing reports:', error);
      throw error;
    }
  },

  // --- Settings ---
  async saveSettings<T>(settings: T): Promise<void> {
    try {
      await SETTINGS_STORE.setItem('settings', settings);
    } catch (error) {
      console.error('[StorageService] Error saving settings:', error);
    }
  },

  async getSettings<T>(): Promise<T | null> {
    try {
      return await SETTINGS_STORE.getItem<T>('settings');
    } catch (error) {
      console.error('[StorageService] Error fetching settings:', error);
      return null;
    }
  },
  
  async clearSettings(): Promise<void> {
    try {
      await SETTINGS_STORE.removeItem('settings');
    } catch (error) {
      console.error('[StorageService] Error clearing settings:', error);
    }
  }
};
