import { toJpeg } from 'html-to-image';
import jsPDF from 'jspdf';
import { toast } from 'sonner';
import { ALL_REGIONS } from '../services/inference/types';
import type { CouncilConsensusResult } from '../services/inference/types';
import type { WorkstationSettings } from '../hooks/useWorkstationSettings';

/**
 * Generates a high-fidelity clinical PDF report.
 * Combines structured data with a visual snapshot of the diagnostic workstation.
 */
export const generatePDFReport = async (
  elementId: string, 
  filename: string,
  data?: CouncilConsensusResult,
  settings?: WorkstationSettings
) => {
  const element = document.getElementById(elementId);
  if (!element) {
    toast.error('Diagnostic error: Target UI layer not found.');
    return;
  }

  const toastId = toast.loading('Sequencing clinical report...');

  try {
    // 1. Capture Visual Evidence
    const dataUrl = await toJpeg(element, {
      quality: 0.95,
      pixelRatio: 2,
      backgroundColor: '#0d1324',
      style: { borderRadius: '0' },
    });

    // 2. Initialize PDF (A4 Portrait)
    const pdf = new jsPDF({
      orientation: 'p',
      unit: 'mm',
      format: 'a4',
      compress: true,
    });

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 20;
    let currY = 20;

    // --- Header Section ---
    pdf.setFillColor(13, 19, 36); // App dark background color
    pdf.rect(0, 0, pageWidth, 40, 'F');
    
    let modalityText = 'MRI (Brain)';
    let subheaderText = 'ADVANCED CLINICAL NEURO-DIAGNOSTICS';
    
    if (elementId === 'ct-report') {
      modalityText = 'CT (Torso)';
      subheaderText = 'ADVANCED CLINICAL TORSO-DIAGNOSTICS';
    } else if (elementId === 'xray-report') {
      modalityText = 'X-Ray (Skeleton)';
      subheaderText = 'ADVANCED CLINICAL OSTEO-DIAGNOSTICS';
    }

    pdf.setTextColor(255, 255, 255);
    pdf.setFontSize(22);
    pdf.setFont('helvetica', 'bold');
    pdf.text('COUNCILMED', margin, 25);
    
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    pdf.text(subheaderText, margin, 32);
    
    pdf.setFontSize(9);
    pdf.text(`REPORT ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()}`, pageWidth - margin - 50, 25);
    pdf.text(`DATE: ${new Date().toLocaleDateString()}`, pageWidth - margin - 50, 32);

    currY = 55;

    // --- Metadata Section ---
    pdf.setTextColor(40, 40, 40);
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'bold');
    pdf.text('CLINICAL METADATA', margin, currY);
    pdf.line(margin, currY + 2, pageWidth - margin, currY + 2);
    
    currY += 10;
    pdf.setFont('helvetica', 'normal');
    pdf.text('Ordering Physician:', margin, currY);
    pdf.text(settings?.doctorName || 'Not Specified', margin + 40, currY);
    
    pdf.text('Clinical ID:', pageWidth / 2, currY);
    pdf.text(settings?.clinicalId || 'N/A', pageWidth / 2 + 40, currY);
    
    currY += 7;
    pdf.text('Specialization:', margin, currY);
    pdf.text(settings?.specialization || 'Radiology', margin + 40, currY);
    
    pdf.text('Modality:', pageWidth / 2, currY);
    pdf.text(modalityText, pageWidth / 2 + 40, currY);

    currY += 15;

    // --- Primary Findings ---
    if (data) {
      pdf.setFillColor(240, 244, 255);
      pdf.rect(margin - 5, currY - 5, pageWidth - (margin * 2) + 10, 35, 'F');
      
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(11);
      pdf.setTextColor(13, 19, 36);
      pdf.text('EXECUTIVE DIAGNOSTIC SUMMARY', margin, currY + 2);
      
      currY += 12;
      pdf.setFontSize(16);
      const severityColor = data.severity === 'critical' ? [220, 38, 38] : data.severity === 'warning' ? [217, 119, 6] : [5, 150, 105];
      pdf.setTextColor(severityColor[0], severityColor[1], severityColor[2]);
      pdf.text(data.primaryDiagnosis.toUpperCase(), margin, currY + 5);
      
      pdf.setFontSize(10);
      pdf.setTextColor(100, 100, 100);
      pdf.text(`Confidence Score: ${data.overallConfidence.toFixed(1)}%`, pageWidth - margin - 50, currY + 5);
      
      currY += 25;
      
      // Detailed Metrics
      pdf.setTextColor(40, 40, 40);
      pdf.setFont('helvetica', 'bold');
      pdf.text('ANATOMICAL LOCALIZATION', margin, currY);
      pdf.line(margin, currY + 2, pageWidth - margin, currY + 2);
      
      currY += 10;
      pdf.setFont('helvetica', 'normal');
      pdf.text('Primary ROI:', margin, currY);
      pdf.text(ALL_REGIONS[data.affectedRegion]?.label || data.affectedRegion, margin + 40, currY);
      
      pdf.text('Model Agreement:', pageWidth / 2, currY);
      pdf.text(`${(data.modelAgreement * 100).toFixed(0)}%`, pageWidth / 2 + 40, currY);
      
      currY += 7;
      if (data.swinResult.tumorVolumeMm3 > 0) {
        pdf.text('Estimated Volume:', margin, currY);
        pdf.text(`${data.swinResult.tumorVolumeMm3.toFixed(1)} mm³`, margin + 40, currY);
      }
      
      pdf.text('Processing Time:', pageWidth / 2, currY);
      pdf.text(`${data.totalInferenceTimeMs.toFixed(0)} ms`, pageWidth / 2 + 40, currY);
      
      currY += 15;
    }

    // --- Visual Evidence Section ---
    pdf.setFont('helvetica', 'bold');
    pdf.text('VISUAL DIAGNOSTIC EVIDENCE', margin, currY);
    pdf.line(margin, currY + 2, pageWidth - margin, currY + 2);
    
    currY += 10;
    const imgWidth = pageWidth - (margin * 2);
    const imgHeight = (imgWidth * 9) / 16; // Maintain aspect ratio
    
    // Check if we need a new page
    if (currY + imgHeight > pageHeight - 30) {
      pdf.addPage();
      currY = 20;
    }
    
    pdf.addImage(dataUrl, 'JPEG', margin, currY, imgWidth, imgHeight, undefined, 'FAST');
    currY += imgHeight + 15;

    // --- Clinical Notes ---
    if (data && data.councilNotes.length > 0) {
      if (currY + 40 > pageHeight - 30) {
        pdf.addPage();
        currY = 20;
      }
      
      pdf.setFont('helvetica', 'bold');
      pdf.text('COUNCIL CLINICAL NOTES', margin, currY);
      pdf.line(margin, currY + 2, pageWidth - margin, currY + 2);
      
      currY += 10;
      pdf.setFont('helvetica', 'italic');
      pdf.setFontSize(9);
      pdf.setTextColor(80, 80, 80);
      
      data.councilNotes.forEach(note => {
        const splitNote = pdf.splitTextToSize(`• ${note}`, pageWidth - (margin * 2));
        pdf.text(splitNote, margin, currY);
        currY += (splitNote.length * 5);
      });
    }

    // --- Footer ---
    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(8);
    pdf.setTextColor(150, 150, 150);
    const footerText = 'This report is AI-generated and intended for clinical decision support. Final diagnosis must be confirmed by a board-certified radiologist.';
    pdf.text(footerText, pageWidth / 2, pageHeight - 15, { align: 'center' });

    // Finalize
    pdf.save(filename);

    toast.success('Clinical Report Finalized', { 
      id: toastId,
      description: `Saved as ${filename}`
    });
  } catch (error) {
    console.error('[PDF Engine Error]:', error);
    toast.error('PDF Generation Failed', { 
       id: toastId,
       description: 'Check if your browser supports WebGL snapshotting.'
    });
  }
};
