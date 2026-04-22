<template>
  <section class="findings-pane">
    <v-list lines="two" class="rounded-lg">
      <v-list-item
        v-for="term in terms"
        :key="term.hpo_id"
        @mouseenter="$emit('hover-term', term.hpo_id)"
        @mouseleave="$emit('clear-hover')"
        @click="$emit('inspect-term', term.hpo_id)"
      >
        <template #title>
          {{ term.name }}
        </template>
        <template #subtitle>
          {{ confidenceBand(term.confidence) }}
        </template>
      </v-list-item>
    </v-list>
  </section>
</template>

<script setup>
defineProps({
  terms: {
    type: Array,
    default: () => [],
  },
});

defineEmits(['hover-term', 'clear-hover', 'inspect-term']);

function confidenceBand(value) {
  if (value >= 0.85) return 'High';
  if (value >= 0.6) return 'Medium';
  return 'Low';
}
</script>
