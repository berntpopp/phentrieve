<template>
  <v-menu :model-value="visible" location="top">
    <v-list density="compact">
      <v-list-item title="Inspect" @click="$emit('inspect')" />
      <v-list-item title="Add to case" @click="$emit('add-to-case')" />
      <v-list-item title="Change term" @click="$emit('change-term')" />
      <v-list-item title="Remove annotation" @click="$emit('remove-annotation')" />
    </v-list>
  </v-menu>
</template>

<script setup>
defineProps({
  visible: {
    type: Boolean,
    default: false,
  },
});

defineEmits(['inspect', 'add-to-case', 'change-term', 'remove-annotation']);
</script>
