<template>
  <div class="similarity-score-wrapper">
    <v-chip
      :color="scoreColor"
      :variant="chipVariant"
      size="small"
      label
      class="score-chip elevation-1"
      :class="{ 'score-chip--pulsing': isPulsing }"
    >
      <v-icon :icon="scoreIcon" size="x-small" start class="score-icon" />
      <span class="score-value">{{ formattedScore }}</span>

      <!-- Accessible tooltip with rich content -->
      <v-tooltip
        v-model="showTooltip"
        activator="parent"
        location="top"
        :open-delay="300"
        max-width="320"
        content-class="score-tooltip-content"
      >
        <div class="score-tooltip">
          <!-- Tooltip header -->
          <div class="tooltip-header">
            <v-icon :icon="scoreIcon" size="small" class="tooltip-icon" />
            <span class="tooltip-title">{{ tooltipTitle }}</span>
          </div>

          <!-- Score quality label -->
          <div class="quality-label" :class="`quality--${qualityLevel}`">
            {{ qualityText }}
          </div>

          <!-- Metric explanation -->
          <div class="tooltip-metric">
            {{ $t('resultsDisplay.cosineSimilarity', 'Cosine Similarity') }}
          </div>

          <!-- Range explanation -->
          <div class="tooltip-range">
            {{
              $t(
                'resultsDisplay.rangeExplanation',
                {
                  min: '0.0',
                  max: '1.0',
                },
                '0.0 (no similarity) to 1.0 (perfect match)'
              )
            }}
          </div>

          <!-- Visual scale with gradient -->
          <div class="scale-visual">
            <div class="scale-bar">
              <div class="scale-fill" :style="scaleFillStyle" />
              <div class="scale-marker" :style="scaleMarkerStyle">
                <div class="marker-dot" />
              </div>
            </div>
            <div class="scale-labels">
              <span class="scale-label-min">
                {{ $t('resultsDisplay.noMatch', 'No Match') }}
              </span>
              <span class="scale-label-max">
                {{ $t('resultsDisplay.perfectMatch', 'Perfect Match') }}
              </span>
            </div>
          </div>

          <!-- Technical note -->
          <div class="tooltip-note">
            <v-icon icon="mdi-information-outline" size="x-small" />
            <span>
              {{ $t('resultsDisplay.scoreNote', 'Represents semantic similarity in vector space') }}
            </span>
          </div>
        </div>
      </v-tooltip>
    </v-chip>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';

/**
 * Props:
 * @property {number} score - The similarity/confidence score (0.0 to 1.0)
 * @property {'similarity'|'rerank'|'confidence'} [type='similarity'] - Type of score being displayed
 * @property {number} [decimals=2] - Number of decimal places to display
 * @property {boolean} [showAnimation=false] - Whether to show pulsing animation
 */
const props = defineProps({
  score: {
    type: Number,
    required: true,
    validator: (value) => value >= 0 && value <= 1,
  },
  type: {
    type: String,
    default: 'similarity',
    validator: (value) => ['similarity', 'rerank', 'confidence'].includes(value),
  },
  decimals: {
    type: Number,
    default: 2,
  },
  showAnimation: {
    type: Boolean,
    default: false,
  },
});

const { t } = useI18n();
const showTooltip = ref(false);

// Format score to specified decimal places
const formattedScore = computed(() => {
  const score = Math.max(0, Math.min(1, props.score)); // Clamp to [0, 1]
  return score.toFixed(props.decimals);
});

// Determine quality level for color coding
const qualityLevel = computed(() => {
  const score = props.score;
  if (score >= 0.9) return 'excellent';
  if (score >= 0.75) return 'high';
  if (score >= 0.6) return 'moderate';
  if (score >= 0.4) return 'low';
  return 'poor';
});

// Score color based on quality (Material Design 3 color palette)
const scoreColor = computed(() => {
  const score = props.score;
  if (score >= 0.9) return 'success';
  if (score >= 0.75) return 'info';
  if (score >= 0.6) return 'warning';
  if (score >= 0.4) return 'orange-darken-2';
  return 'error';
});

// Chip variant for visual hierarchy
const chipVariant = computed(() => {
  return props.score >= 0.75 ? 'elevated' : 'flat';
});

// Dynamic icon based on score quality
const scoreIcon = computed(() => {
  const score = props.score;
  if (props.type === 'rerank') return 'mdi-sort-ascending';
  if (props.type === 'confidence') return 'mdi-shield-check';

  // For similarity scores
  if (score >= 0.9) return 'mdi-star';
  if (score >= 0.75) return 'mdi-chart-line-variant';
  if (score >= 0.6) return 'mdi-chart-line';
  return 'mdi-chart-timeline-variant';
});

// Quality text labels
const qualityText = computed(() => {
  const level = qualityLevel.value;
  const key = `resultsDisplay.quality.${level}`;
  const defaults = {
    excellent: 'Excellent Match',
    high: 'High Match',
    moderate: 'Moderate Match',
    low: 'Low Match',
    poor: 'Poor Match',
  };
  return t(key, defaults[level]);
});

// Tooltip title based on type
const tooltipTitle = computed(() => {
  const typeLabels = {
    similarity: t('resultsDisplay.similarityLabel', 'Similarity Score'),
    rerank: t('resultsDisplay.rerankLabel', 'Re-ranking Score'),
    confidence: t('resultsDisplay.confidenceLabel', 'Confidence Score'),
  };
  return typeLabels[props.type];
});

// Scale fill style with gradient
const scaleFillStyle = computed(() => {
  const percentage = props.score * 100;
  return {
    width: `${percentage}%`,
    transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
  };
});

// Scale marker position
const scaleMarkerStyle = computed(() => {
  const percentage = props.score * 100;
  return {
    left: `${percentage}%`,
    transition: 'left 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
  };
});

// Pulsing animation for high scores
const isPulsing = computed(() => {
  return props.showAnimation && props.score >= 0.85;
});
</script>

<style scoped>
.similarity-score-wrapper {
  display: inline-block;
}

/* Score chip styling */
.score-chip {
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.025em;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: help;
  user-select: none;
}

.score-chip:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
}

.score-value {
  font-weight: 600;
  font-size: 0.875rem;
  letter-spacing: 0.5px;
  font-family: 'Roboto Mono', monospace;
}

.score-icon {
  opacity: 0.9;
}

/* Pulsing animation for excellent scores */
@keyframes pulse {
  0%,
  100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.03);
    opacity: 0.95;
  }
}

.score-chip--pulsing {
  animation: pulse 2s ease-in-out infinite;
}

/* Tooltip content styling */
.score-tooltip {
  padding: 12px;
  font-size: 13px;
  line-height: 1.5;
}

.tooltip-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.15);
}

.tooltip-icon {
  opacity: 0.9;
}

.tooltip-title {
  font-weight: 600;
  font-size: 14px;
  color: rgba(255, 255, 255, 0.95);
}

/* Quality label styling */
.quality-label {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
}

.quality--excellent {
  background: rgba(76, 175, 80, 0.25);
  color: #a5d6a7;
  border: 1px solid rgba(76, 175, 80, 0.3);
}

.quality--high {
  background: rgba(33, 150, 243, 0.25);
  color: #90caf9;
  border: 1px solid rgba(33, 150, 243, 0.3);
}

.quality--moderate {
  background: rgba(255, 193, 7, 0.25);
  color: #ffe082;
  border: 1px solid rgba(255, 193, 7, 0.3);
}

.quality--low {
  background: rgba(255, 152, 0, 0.25);
  color: #ffcc80;
  border: 1px solid rgba(255, 152, 0, 0.3);
}

.quality--poor {
  background: rgba(244, 67, 54, 0.25);
  color: #ef9a9a;
  border: 1px solid rgba(244, 67, 54, 0.3);
}

/* Metric and range text */
.tooltip-metric {
  color: rgba(255, 255, 255, 0.85);
  font-size: 12px;
  margin-bottom: 4px;
  font-weight: 500;
}

.tooltip-range {
  color: rgba(255, 255, 255, 0.7);
  font-size: 11px;
  margin-bottom: 12px;
}

/* Visual scale styling */
.scale-visual {
  margin-top: 12px;
  margin-bottom: 10px;
}

.scale-bar {
  position: relative;
  height: 8px;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 4px;
  overflow: visible;
  margin-bottom: 6px;
}

.scale-fill {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: linear-gradient(
    to right,
    #ef5350 0%,
    #ff9800 25%,
    #ffc107 40%,
    #66bb6a 60%,
    #4caf50 100%
  );
  border-radius: 4px;
  box-shadow: 0 0 8px rgba(76, 175, 80, 0.4);
}

.scale-marker {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 2;
}

.marker-dot {
  width: 12px;
  height: 12px;
  background: white;
  border: 2px solid rgba(0, 0, 0, 0.2);
  border-radius: 50%;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.scale-labels {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: rgba(255, 255, 255, 0.6);
  margin-top: 6px;
  letter-spacing: 0.3px;
}

.scale-label-min,
.scale-label-max {
  font-weight: 500;
}

/* Technical note */
.tooltip-note {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  font-size: 10px;
  color: rgba(255, 255, 255, 0.65);
  line-height: 1.4;
}

.tooltip-note v-icon {
  margin-top: 1px;
  flex-shrink: 0;
}

/* Responsive adjustments */
@media (max-width: 600px) {
  .score-chip {
    font-size: 0.8rem;
  }

  .score-value {
    font-size: 0.8rem;
  }

  :deep(.score-tooltip-content) {
    max-width: 280px !important;
  }
}

/* Dark theme enhancements */
@media (prefers-color-scheme: dark) {
  .score-chip {
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  }
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .score-chip {
    border: 2px solid currentColor;
  }

  .scale-fill {
    box-shadow: none;
    border: 1px solid rgba(255, 255, 255, 0.5);
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .score-chip,
  .scale-fill,
  .scale-marker {
    transition: none;
  }

  .score-chip--pulsing {
    animation: none;
  }
}
</style>
