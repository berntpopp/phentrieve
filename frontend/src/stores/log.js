import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useLogStore = defineStore('log', () => {
  const logs = ref([])
  const isViewerVisible = ref(false)

  function addLogEntry(entry) {
    logs.value.push(entry)
  }

  function clearLogs() {
    logs.value = []
  }

  function trimLogs(maxEntries) {
    if (logs.value.length > maxEntries) {
      logs.value = logs.value.slice(-maxEntries)
    }
  }

  function setViewerVisibility(visible) {
    isViewerVisible.value = visible
  }

  function toggleViewer() {
    isViewerVisible.value = !isViewerVisible.value
  }

  return {
    logs,
    isViewerVisible,
    addLogEntry,
    clearLogs,
    trimLogs,
    setViewerVisibility,
    toggleViewer
  }
})
