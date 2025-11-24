<script setup>
/**
 * ConversationSkeleton Component
 *
 * Displays shimmer loading placeholders for conversation history.
 * Uses GPU-accelerated CSS animations for smooth 60fps performance.
 * Pattern inspired by Facebook/LinkedIn skeleton loading.
 *
 * @component
 */
defineProps({
  /**
   * Number of skeleton items to display
   */
  count: {
    type: Number,
    default: 2,
  },
});
</script>

<template>
  <div class="conversation-skeleton" role="status" aria-label="Loading conversation history">
    <!-- Vue iterates 1 to count when given a number -->
    <div v-for="index in count" :key="index" class="skeleton-item">
      <!-- Query skeleton -->
      <div class="skeleton-query">
        <div class="skeleton-avatar shimmer" />
        <div class="skeleton-content">
          <div class="skeleton-line skeleton-line--short shimmer" />
          <div class="skeleton-line skeleton-line--long shimmer" />
        </div>
      </div>

      <!-- Response skeleton -->
      <div class="skeleton-response">
        <div class="skeleton-card">
          <div class="skeleton-header">
            <div class="skeleton-badge shimmer" />
            <div class="skeleton-line skeleton-line--medium shimmer" />
          </div>
          <div class="skeleton-body">
            <div class="skeleton-row">
              <div class="skeleton-chip shimmer" />
              <div class="skeleton-chip shimmer" />
              <div class="skeleton-chip shimmer" />
            </div>
            <div class="skeleton-line skeleton-line--full shimmer" />
            <div class="skeleton-line skeleton-line--medium shimmer" />
          </div>
        </div>
      </div>
    </div>

    <!-- Screen reader text -->
    <span class="sr-only">Loading conversation history...</span>
  </div>
</template>

<style scoped>
.conversation-skeleton {
  padding: 16px;
}

.skeleton-item {
  margin-bottom: 24px;
}

.skeleton-item:last-child {
  margin-bottom: 0;
}

/* Query skeleton styles */
.skeleton-query {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 16px;
}

.skeleton-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #e0e0e0;
  flex-shrink: 0;
}

.skeleton-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* Response skeleton styles */
.skeleton-response {
  margin-left: 52px;
}

.skeleton-card {
  background: #f5f5f5;
  border-radius: 8px;
  padding: 16px;
}

.skeleton-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.skeleton-badge {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  background: #e0e0e0;
}

.skeleton-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.skeleton-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.skeleton-chip {
  width: 80px;
  height: 28px;
  border-radius: 14px;
  background: #e0e0e0;
}

.skeleton-chip:nth-child(2) {
  width: 100px;
}

.skeleton-chip:nth-child(3) {
  width: 70px;
}

/* Skeleton line variations */
.skeleton-line {
  height: 12px;
  border-radius: 6px;
  background: #e0e0e0;
}

.skeleton-line--short {
  width: 30%;
}

.skeleton-line--medium {
  width: 60%;
}

.skeleton-line--long {
  width: 85%;
}

.skeleton-line--full {
  width: 100%;
}

/* Shimmer animation - GPU accelerated */
.shimmer {
  position: relative;
  overflow: hidden;
}

.shimmer::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.4) 50%,
    transparent 100%
  );
  transform: translateX(-100%);
  animation: shimmer 1.5s infinite ease-in-out;
  will-change: transform;
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

/* Screen reader only text */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* Dark mode support via Vuetify theme */
:root.v-theme--dark .skeleton-avatar,
:root.v-theme--dark .skeleton-badge,
:root.v-theme--dark .skeleton-chip,
:root.v-theme--dark .skeleton-line {
  background: #424242;
}

:root.v-theme--dark .skeleton-card {
  background: #303030;
}

:root.v-theme--dark .shimmer::after {
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.1) 50%,
    transparent 100%
  );
}
</style>
