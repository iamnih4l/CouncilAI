import { motion } from 'framer-motion';
import type { CouncilConsensusResult } from '../../services/inference/types';

interface ImageViewer2DProps {
  imageSrc: string;
  consensusResult?: CouncilConsensusResult | null;
}

export function ImageViewer2D({ imageSrc, consensusResult }: ImageViewer2DProps) {
  // Extract bounding box metrics
  const getOverlayStyles = () => {
    if (!consensusResult || !consensusResult.attentionResult) return null;

    const { focusCenter, focusRadius } = consensusResult.attentionResult;
    
    // Map from [-3, 3] 3D coordinates back to 0-100% 2D image percentages
    // attentionNet output: mappedX = ((maxHeatX / heatmapSize) * 6) - 3
    // attentionNet output: mappedY = -(((maxHeatY / heatmapSize) * 6) - 3)
    const xPercent = ((focusCenter[0] + 3) / 6) * 100;
    const yPercent = ((-focusCenter[1] + 3) / 6) * 100;
    
    // Radius mapping
    const radiusPercent = (focusRadius / 6) * 100;
    const diameterPercent = radiusPercent * 2;

    // Determine color based on severity
    let borderColor = 'border-emerald-500';
    let bgColor = 'bg-emerald-500/20';
    
    switch (consensusResult.severity) {
      case 'critical':
        borderColor = 'border-red-500';
        bgColor = 'bg-red-500/20';
        break;
      case 'warning':
        borderColor = 'border-amber-500';
        bgColor = 'bg-amber-500/20';
        break;
      case 'info':
        borderColor = 'border-cyan-500';
        bgColor = 'bg-cyan-500/20';
        break;
    }

    return {
      xPercent,
      yPercent,
      diameterPercent,
      borderColor,
      bgColor,
    };
  };

  const overlay = getOverlayStyles();

  return (
    <div className="relative w-full h-full flex items-center justify-center p-4 bg-[#0a0a0a] rounded-lg overflow-hidden">
      <div className="relative max-w-full max-h-full">
        {/* The Base Scan Image */}
        <img 
          src={imageSrc} 
          alt="Patient Scan" 
          className="max-w-full max-h-full object-contain rounded shadow-2xl" 
          style={{ maxHeight: '60vh' }}
        />

        {/* The AI Anomaly Overlay */}
        {overlay && (
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', bounce: 0.5 }}
            className={`absolute rounded-full border-2 ${overlay.borderColor} ${overlay.bgColor}`}
            style={{
              left: `${overlay.xPercent}%`,
              top: `${overlay.yPercent}%`,
              width: `${overlay.diameterPercent}%`,
              height: `${overlay.diameterPercent}%`,
              transform: 'translate(-50%, -50%)',
              boxShadow: '0 0 20px rgba(0,0,0,0.5)',
            }}
          >
            {/* Crosshair indicator */}
            <div className="absolute top-1/2 left-1/2 w-2 h-2 rounded-full bg-white transform -translate-x-1/2 -translate-y-1/2 opacity-75 shadow-lg" />
          </motion.div>
        )}
      </div>
    </div>
  );
}
