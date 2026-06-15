<!-- src/views/McpAccess.vue -->
<!--
  Connect an AI Agent (MCP) page.

  Explains how to connect Phentrieve's public, read-only Model Context Protocol
  (MCP) server to AI clients: Claude (web + desktop), ChatGPT developer mode,
  and coding agents (Claude Code, Codex CLI).

  The Phentrieve MCP server is served same-origin under `/mcp` (the frontend
  Nginx proxies it to the API). That is why this informational page lives at
  `/connect` rather than `/mcp`: anything under `/mcp` is the MCP transport
  endpoint for AI clients, not a website that renders in a browser.
-->
<template>
  <v-container class="py-8">
    <v-row justify="center">
      <v-col cols="12" md="10" lg="9">
        <!-- Back to home — mirrors the FAQ view's affordance; ms-n2 optically
             aligns the icon with the content edge below it. -->
        <v-btn
          variant="text"
          color="primary"
          class="mb-4 ms-n2"
          :to="{ name: 'home' }"
          prepend-icon="mdi-arrow-left"
          aria-label="Navigate back to home page"
        >
          Back to home
        </v-btn>

        <!-- Header -->
        <div class="text-center mb-8">
          <v-avatar color="primary" size="80" class="mb-4">
            <v-icon size="48" color="white">mdi-robot-outline</v-icon>
          </v-avatar>
          <h1 class="text-h3 font-weight-bold text-high-emphasis mb-2">
            Connect an AI Agent (MCP)
          </h1>
          <p class="text-h6 text-medium-emphasis">
            Map clinical text to HPO terms from Claude, ChatGPT, and coding agents over the Model
            Context Protocol
          </p>
        </div>

        <!-- Page-vs-endpoint notice -->
        <v-alert type="info" variant="tonal" class="mb-6" icon="mdi-information-outline">
          This page explains how to connect. The address below is an MCP
          <strong>transport endpoint for AI clients</strong> — not a website, so opening it in a
          browser will not show a page.
        </v-alert>

        <!-- What you get -->
        <v-card class="mb-6" elevation="2">
          <v-card-title class="card-band">
            <v-icon color="primary" class="mr-2">mdi-database-search-outline</v-icon>
            What you get
          </v-card-title>
          <v-card-text class="pa-6">
            <p class="text-body-1 mb-4">
              A <strong>read-only</strong> connection to Phentrieve's HPO term retrieval and
              annotation tools. Your agent can search candidate HPO terms, extract phenotypes from
              research text (deterministic or LLM-assisted), compare term similarity, chunk long
              documents, and export GA4GH Phenopackets — no write access, no sign-in, and no API
              key.
            </p>
            <div class="d-flex flex-wrap ga-2 mb-4">
              <v-chip
                v-for="tool in tools"
                :key="tool"
                size="small"
                variant="tonal"
                color="primary"
                label
              >
                <v-icon start size="14">mdi-tools</v-icon>
                {{ tool }}
              </v-chip>
            </div>
            <p class="text-body-2 text-medium-emphasis mb-0">
              Research use only — Phentrieve's outputs are algorithmic research suggestions, not
              clinical decision support. Do not submit identifiable patient data to public
              instances.
            </p>
          </v-card-text>
        </v-card>

        <!-- Server address -->
        <v-card class="mb-6" elevation="2">
          <v-card-title class="card-band">
            <v-icon color="primary" class="mr-2">mdi-link-variant</v-icon>
            Server address
          </v-card-title>
          <v-card-text class="pa-6">
            <div class="d-flex align-center flex-wrap ga-3">
              <code class="endpoint-code">{{ mcpEndpoint }}</code>
              <v-btn
                size="small"
                variant="tonal"
                color="primary"
                prepend-icon="mdi-content-copy"
                @click="copy(mcpEndpoint, 'Server address')"
              >
                Copy
              </v-btn>
              <v-btn
                size="small"
                variant="text"
                color="primary"
                prepend-icon="mdi-api"
                :href="apiDocsUrl"
                target="_blank"
                rel="noopener noreferrer"
              >
                REST API docs
              </v-btn>
            </div>
            <p class="text-body-2 text-medium-emphasis mt-3 mb-0">
              Transport: MCP Streamable HTTP. Use this same address in every client below. Some
              clients need a trailing slash (<code>{{ mcpEndpoint }}/</code>) if they do not follow
              HTTP redirects.
            </p>
          </v-card-text>
        </v-card>

        <!-- Per-client instructions -->
        <h2 class="text-h5 font-weight-bold text-high-emphasis mb-4">How to connect</h2>
        <v-row>
          <v-col v-for="client in clients" :key="client.id" cols="12" md="6">
            <v-card class="h-100 d-flex flex-column" elevation="2">
              <v-card-title class="d-flex align-center">
                <v-icon :color="client.color" class="mr-2">{{ client.icon }}</v-icon>
                {{ client.name }}
              </v-card-title>
              <v-card-text class="flex-grow-1">
                <ol class="client-steps mb-3">
                  <li v-for="(step, i) in client.steps" :key="i" class="mb-1">{{ step }}</li>
                </ol>
                <div class="snippet-wrap">
                  <pre class="snippet"><code>{{ client.snippet }}</code></pre>
                  <v-btn
                    class="snippet-copy"
                    size="x-small"
                    variant="text"
                    icon="mdi-content-copy"
                    :aria-label="`Copy ${client.name} configuration`"
                    @click="copy(client.snippet, `${client.name} configuration`)"
                  />
                </div>
                <p v-if="client.note" class="text-caption text-medium-emphasis mt-2 mb-0">
                  {{ client.note }}
                </p>
              </v-card-text>
            </v-card>
          </v-col>
        </v-row>

        <!-- Verify tip -->
        <v-alert type="success" variant="tonal" class="mt-6" icon="mdi-check-circle-outline">
          Once connected, ask your agent to call <code>phentrieve_get_capabilities</code> first — it
          lists every available tool, the response modes, limits, error codes, and the citation
          contract.
        </v-alert>

        <p class="text-caption text-medium-emphasis mt-4 mb-0">
          New to MCP? See the
          <a href="https://modelcontextprotocol.io" target="_blank" rel="noopener noreferrer">
            Model Context Protocol
          </a>
          documentation, or the
          <a
            href="https://berntpopp.github.io/phentrieve/"
            target="_blank"
            rel="noopener noreferrer"
          >
            Phentrieve project docs </a
          >.
        </p>
      </v-col>
    </v-row>

    <!-- Copy confirmation -->
    <v-snackbar v-model="snackbar" :timeout="2000" :color="snackbarSuccess ? 'success' : 'error'">
      {{ snackbarMessage }}
    </v-snackbar>
  </v-container>
</template>

<script setup>
import { ref, computed } from 'vue';
import { API_URL } from '@/services/apiClient';
import { logService } from '@/services/logService';

// Phentrieve serves MCP same-origin under `/mcp` (Nginx proxies it to the API),
// so derive the endpoint from the current origin. This adapts automatically to
// production (https://phentrieve.org/mcp), staging, and the local Docker stack
// (http://localhost:8080/mcp). VITE_MCP_URL overrides it for custom deployments.
const origin = typeof window !== 'undefined' ? window.location.origin : 'https://phentrieve.org';
const mcpEndpoint = ref((import.meta.env.VITE_MCP_URL || `${origin}/mcp`).replace(/\/+$/, ''));

// Live OpenAPI (Swagger UI) docs URL, derived from the configured API base.
const apiDocsUrl = `${API_URL.replace(/\/+$/, '')}/docs`;

// Read-only tools exposed by the server (kept in sync with docs/mcp-server.md).
const tools = [
  'phentrieve_search_hpo_terms',
  'phentrieve_extract_hpo_terms',
  'phentrieve_extract_hpo_terms_llm',
  'phentrieve_compare_hpo_terms',
  'phentrieve_chunk_text',
  'phentrieve_export_phenopacket',
  'phentrieve_get_capabilities',
  'phentrieve_diagnostics',
];

// Connection recipes per client, verified against current (2026) official docs.
// Steps are intentionally short; the copy-pasteable command/config lives in
// `snippet`, and `note` carries the most important caveat. Endpoint and server
// name are interpolated so the snippets are correct for this deployment.
const serverName = 'phentrieve';
const clients = computed(() => {
  const url = mcpEndpoint.value;
  return [
    {
      id: 'claude',
      name: 'Claude (web & desktop)',
      icon: 'mdi-creation',
      color: 'deep-orange',
      steps: [
        'In Claude, open Settings → Connectors (claude.ai/settings/connectors).',
        'Click "Add custom connector".',
        'Paste the server address into the URL field and click Add (leave the OAuth fields empty).',
        'In a chat, open the "+" menu → Connectors and toggle phentrieve on.',
      ],
      snippet: url,
      note: 'Same steps on claude.ai and Claude Desktop. Custom connectors are in beta; the Free plan allows one.',
    },
    {
      id: 'chatgpt',
      name: 'ChatGPT',
      icon: 'mdi-chat-processing-outline',
      color: 'green-darken-1',
      steps: [
        'On chatgpt.com, open Settings → Apps & Connectors → Advanced and turn on Developer mode.',
        'Click "Create" to add a new connector.',
        `Set Name to "Phentrieve", MCP server URL to the address, Authentication to "No authentication", then create.`,
        'In a chat, open the "+" menu and enable the Phentrieve app.',
      ],
      snippet: `Name:  Phentrieve\nURL:   ${url}\nAuth:  No authentication`,
      note: 'Web only (chatgpt.com); Plus, Pro, Business, Enterprise, or Edu. Developer mode is in beta.',
    },
    {
      id: 'claude-code',
      name: 'Claude Code',
      icon: 'mdi-console',
      color: 'blue-grey-darken-1',
      steps: [
        'Run the command below (flags go before the server name).',
        'For a shared/team setup, add --scope project to write it into .mcp.json.',
        'Verify with "claude mcp list", or run "/mcp" inside a session.',
      ],
      snippet: `claude mcp add --transport http ${serverName} ${url}`,
      note: `Or hand-edit .mcp.json → { "mcpServers": { "${serverName}": { "type": "http", "url": "${url}" } } }`,
    },
    {
      id: 'codex',
      name: 'Codex CLI',
      icon: 'mdi-code-braces',
      color: 'indigo-darken-1',
      steps: [
        'Add the table below to ~/.codex/config.toml (or run the shortcut command).',
        'Verify with "codex mcp list", then start Codex.',
        'If it does not connect, update Codex to the latest release first.',
      ],
      snippet: `# ~/.codex/config.toml\n[mcp_servers.${serverName}]\nurl = "${url}"\n\n# or, on current Codex releases:\ncodex mcp add ${serverName} --url ${url}`,
      note: 'Remote Streamable-HTTP servers are first-class in current Codex releases.',
    },
  ];
});

// Copy confirmation snackbar state.
const snackbar = ref(false);
const snackbarSuccess = ref(true);
const snackbarMessage = ref('');

const showSnackbar = (message, success) => {
  snackbarMessage.value = message;
  snackbarSuccess.value = success;
  snackbar.value = true;
};

/**
 * Copy text to the clipboard and surface the outcome via the snackbar.
 * Uses the async Clipboard API (available in secure contexts); failures and
 * unsupported environments report an error rather than failing silently.
 */
const copy = (text, label) => {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        showSnackbar(`${label} copied to clipboard`, true);
        logService.debug('MCP page clipboard copy', { label });
      })
      .catch((err) => {
        showSnackbar('Failed to copy — select and copy manually', false);
        logService.warn('MCP page clipboard copy failed', { label, error: err.message });
      });
  } else {
    showSnackbar('Copying is unavailable — select and copy manually', false);
    logService.warn('MCP page clipboard API unavailable', { label });
  }
};
</script>

<style scoped>
/* Theme-aware "code surface": faint on-surface tint that follows light/dark. */
.card-band {
  background-color: rgba(var(--v-theme-primary), 0.08);
}

.endpoint-code {
  padding: 0.35rem 0.6rem;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.12);
  border-radius: 6px;
  background: rgba(var(--v-theme-on-surface), 0.04);
  color: rgb(var(--v-theme-on-surface));
  font-size: 0.95rem;
  word-break: break-all;
}

.client-steps {
  padding-left: 1.15rem;
  font-size: 0.92rem;
  line-height: 1.5;
}

.snippet-wrap {
  position: relative;
}

.snippet {
  overflow-x: auto;
  margin: 0;
  padding: 0.75rem 2.25rem 0.75rem 0.85rem;
  border: 1px solid rgba(var(--v-theme-on-surface), 0.12);
  border-radius: 6px;
  background: rgba(var(--v-theme-on-surface), 0.04);
  color: rgb(var(--v-theme-on-surface));
  font-size: 0.82rem;
  line-height: 1.45;
  white-space: pre;
}

.snippet-copy {
  position: absolute;
  top: 0.3rem;
  right: 0.3rem;
}
</style>
