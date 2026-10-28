import { nextTick as vueNextTick } from 'vue';

function getFirstQueryValue(value) {
  return Array.isArray(value) ? value[0] : value;
}

function getStringQueryValue(value) {
  const normalizedValue = getFirstQueryValue(value);
  return typeof normalizedValue === 'string' ? normalizedValue : undefined;
}

function hasOwnQueryParam(query, key) {
  return Object.prototype.hasOwnProperty.call(query ?? {}, key);
}

function getSanitizedSubmissionLog({
  mode,
  text,
  selectedLanguage,
  textProcessOptions,
  piiScanResult = null,
  redactionApplied = false,
}) {
  return {
    mode,
    textLength: String(text ?? '').length,
    selectedLanguage,
    extractionBackend:
      mode === 'textProcess' ? textProcessOptions?.extractionBackend || null : null,
    piiSummary: piiScanResult?.summary || { high: {}, review: {} },
    piiRedactionApplied: redactionApplied,
  };
}

function clearAutoSubmitQueryParamIfNeeded(route, router, isAutoSubmit, logService) {
  if (!isAutoSubmit && hasOwnQueryParam(route?.query, 'autoSubmit')) {
    const newRouteQuery = { ...route.query };
    delete newRouteQuery.autoSubmit;

    Promise.resolve(router.replace({ query: newRouteQuery })).catch((error) => {
      if (error.name !== 'NavigationDuplicated' && error.name !== 'NavigationCancelled') {
        logService.warn('Error clearing autoSubmit from URL:', error);
      }
    });
  }
}

/**
 * Model bootstrap, URL-parameter handling and query submission for QueryInterface.
 *
 * The controller reads and writes the component's refs directly (`.value`), so it
 * can be exercised in tests with plain refs and a fake store/router.
 *
 * @param {Object} options
 * @param {Object} options.state - Refs: queryText, selectedModel, availableModels,
 *   modelsLoading, isLoading, shouldScrollToTop, userHasScrolled,
 *   showAdvancedOptions, forceEndpointMode, isTextProcessModeActive (computed),
 *   selectedLanguage, similarityThreshold, numResults, includeDetails,
 *   expandedUserNotes, textProcessOptions, chunkingStrategy, windowSize, stepSize,
 *   splitThreshold, minSegmentLength, semanticModelForChunking,
 *   retrievalModelForTextProcess, chunkRetrievalThreshold, numResultsPerChunk,
 *   noAssertionDetectionForTextProcess, assertionPreferenceForTextProcess,
 *   aggregatedTermConfidence, topTermPerChunkForAggregation.
 *   Optional: uiLocale (ref of the UI locale, used for PII rules).
 * @param {Object} options.conversationStore - Conversation Pinia store.
 * @param {Object} options.piiReviewFlow - usePiiReviewFlow() instance.
 * @param {Object} options.route - Current route location (reads `query`).
 * @param {Object} options.router - Router (uses `replace`).
 * @param {Function} [options.nextTick] - Defaults to Vue's nextTick.
 * @param {Object} options.service - PhentrieveService-compatible API client.
 * @param {Object} options.logService - Logger.
 */
export function useQueryInterfaceController({
  state,
  conversationStore,
  piiReviewFlow,
  route,
  router,
  nextTick = vueNextTick,
  service,
  logService,
}) {
  let hasAppliedInitialUrlParameters = false;
  let activeRequest = null;

  function setFallbackModel() {
    state.availableModels.value = [
      { text: 'BioLORD 2023-M', value: 'FremyCompany/BioLORD-2023-M' },
    ];
    state.selectedModel.value = 'FremyCompany/BioLORD-2023-M';

    logService.info('Using fallback embedding model', {
      model: state.selectedModel.value,
    });
  }

  function applyUrlParametersAndAutoSubmit() {
    const queryParams = route?.query ?? {};

    logService.debug('URL query parameters received', {
      keys: Object.keys(queryParams),
      hasText: hasOwnQueryParam(queryParams, 'text'),
      textLength: String(getStringQueryValue(queryParams.text) ?? '').length,
    });

    let advancedOptionsWereSet = false;
    const parseBooleanParam = (value) =>
      typeof value === 'string' && (value.toLowerCase() === 'true' || value === '1');

    const textParam = getStringQueryValue(queryParams.text);
    const modelParam = getStringQueryValue(queryParams.model);
    const thresholdParam = getStringQueryValue(queryParams.threshold);
    const forceEndpointModeParam = getStringQueryValue(queryParams.forceEndpointMode);
    const chunkingStrategyParam = getStringQueryValue(queryParams.chunkingStrategy);
    const autoSubmitParam = getStringQueryValue(queryParams.autoSubmit);

    if (textParam !== undefined) {
      state.queryText.value = textParam;
    }

    if (modelParam !== undefined) {
      const validModels = state.availableModels.value.map((model) => model.value);
      if (validModels.includes(modelParam)) {
        state.selectedModel.value = modelParam;
        advancedOptionsWereSet = true;
      }
    }

    if (thresholdParam !== undefined) {
      const parsedThreshold = parseFloat(thresholdParam);
      if (!Number.isNaN(parsedThreshold) && parsedThreshold >= 0 && parsedThreshold <= 1) {
        state.similarityThreshold.value = parsedThreshold;
        advancedOptionsWereSet = true;
      }
    }

    if (forceEndpointModeParam !== undefined) {
      state.forceEndpointMode.value = forceEndpointModeParam;
      advancedOptionsWereSet = true;
    }

    if (chunkingStrategyParam !== undefined) {
      state.chunkingStrategy.value = chunkingStrategyParam;
      advancedOptionsWereSet = true;
    }

    if (advancedOptionsWereSet) {
      state.showAdvancedOptions.value = true;
    }

    const currentQueryText = state.queryText.value;
    const performAutoSubmit =
      autoSubmitParam !== undefined ? parseBooleanParam(autoSubmitParam) : textParam !== undefined;

    if (performAutoSubmit && typeof currentQueryText === 'string' && currentQueryText.trim()) {
      logService.info('Auto-submitting query based on URL parameters.');
      nextTick(() => setTimeout(() => submitQuery(true), 300));
    }
  }

  async function fetchAvailableModels() {
    state.modelsLoading.value = true;

    try {
      const config = await service.getConfigInfo();
      const availableEmbeddingModels = Array.isArray(config?.available_embedding_models)
        ? config.available_embedding_models
        : [];

      if (availableEmbeddingModels.length > 0) {
        const availableModels = availableEmbeddingModels.map((model) => ({
          text: model.id.split('/').pop(),
          value: model.id,
        }));
        const selectedModel = config.default_embedding_model || availableModels[0]?.value || null;

        state.availableModels.value = availableModels;
        state.selectedModel.value = selectedModel;

        logService.info('Loaded embedding models from API', {
          count: availableModels.length,
          defaultModel: selectedModel,
        });
      } else {
        setFallbackModel();
      }
    } catch (error) {
      logService.warn('Failed to fetch models from API, using fallback', {
        error: error instanceof Error ? error.message : String(error),
      });
      setFallbackModel();
    } finally {
      state.modelsLoading.value = false;

      if (!hasAppliedInitialUrlParameters) {
        hasAppliedInitialUrlParameters = true;
        applyUrlParametersAndAutoSubmit();
      }
    }
  }

  async function submitQueryText({
    currentQuery,
    rawQueryForHistory = currentQuery,
    redactedQueryForHistory = null,
    useTextProcessMode,
    isAutoSubmit,
    piiScanResult = null,
    redactionApplied = false,
  }) {
    state.isLoading.value = true;
    state.shouldScrollToTop.value = true;
    state.userHasScrolled.value = false;

    const queryId = conversationStore.addQuery({
      query: rawQueryForHistory,
      redactedQuery: redactedQueryForHistory,
      loading: true,
      type: useTextProcessMode ? 'textProcess' : 'query',
    });

    if (useTextProcessMode) {
      state.expandedUserNotes.value = { ...state.expandedUserNotes.value, [queryId]: true };
    }

    if (!isAutoSubmit) {
      state.queryText.value = '';
    }

    const abortController = new AbortController();
    activeRequest = abortController;
    const requestOptions = { signal: abortController.signal };

    try {
      let response;
      const textProcessOptions = state.textProcessOptions.value;
      const sanitizedLog = getSanitizedSubmissionLog({
        mode: useTextProcessMode ? 'textProcess' : 'query',
        text: currentQuery,
        selectedLanguage: state.selectedLanguage.value,
        textProcessOptions,
        piiScanResult,
        redactionApplied,
      });

      if (useTextProcessMode) {
        const textProcessPayload = {
          text: currentQuery,
          extractionBackend: textProcessOptions.extractionBackend,
          llmModel: textProcessOptions.llmModel,
          llmMode: textProcessOptions.llmMode,
          language: state.selectedLanguage.value,
          chunkingStrategy: state.chunkingStrategy.value,
          windowSize: state.windowSize.value,
          stepSize: state.stepSize.value,
          splitThreshold: state.splitThreshold.value,
          minSegmentLength: state.minSegmentLength.value,
          semanticModelForChunking:
            state.semanticModelForChunking.value || state.selectedModel.value,
          retrievalModelForTextProcess:
            state.retrievalModelForTextProcess.value || state.selectedModel.value,
          chunkRetrievalThreshold: state.chunkRetrievalThreshold.value,
          numResultsPerChunk: state.numResultsPerChunk.value,
          noAssertionDetectionForTextProcess: state.noAssertionDetectionForTextProcess.value,
          assertionPreferenceForTextProcess: state.assertionPreferenceForTextProcess.value,
          aggregatedTermConfidence: state.aggregatedTermConfidence.value,
          topTermPerChunkForAggregation: state.topTermPerChunkForAggregation.value,
          includeDetails: state.includeDetails.value,
        };
        logService.info('Submitting text request', sanitizedLog);
        response = await service.processText(textProcessPayload, requestOptions);
      } else {
        const queryPayload = {
          text: currentQuery,
          model_name: state.selectedModel.value,
          language: state.selectedLanguage.value,
          num_results: state.numResults.value,
          similarity_threshold: state.similarityThreshold.value,
          query_assertion_language: state.selectedLanguage.value,
          detect_query_assertion: true,
          include_details: state.includeDetails.value,
        };
        logService.info('Submitting text request', sanitizedLog);
        response = await service.queryHpo(queryPayload, requestOptions);
      }

      conversationStore.updateQueryResponse(queryId, response);
    } catch (error) {
      if (error?.type === 'CANCELED') {
        // A cancelled request leaves no error turn behind
        const history = conversationStore.queryHistory;
        const index = history.findIndex((item) => item.id === queryId);
        if (index !== -1) history.splice(index, 1);
      } else {
        conversationStore.updateQueryResponse(queryId, null, error);
        logService.error('Error submitting query/processing text', error);
      }
      // Give the text back on failure so a pasted note is not lost (unless the user typed anew)
      if (!isAutoSubmit && !state.queryText.value) {
        state.queryText.value = currentQuery;
      }
    } finally {
      if (activeRequest === abortController) activeRequest = null;
      state.isLoading.value = false;
      clearAutoSubmitQueryParamIfNeeded(route, router, isAutoSubmit, logService);
    }
  }

  async function submitQuery(isAutoSubmit = false) {
    const queryText = state.queryText.value;
    const queryTextTrimmed = typeof queryText === 'string' ? queryText.trim() : '';

    if (!queryTextTrimmed) {
      logService.warn('Empty query submission prevented');
      return;
    }

    const useTextProcessMode = state.isTextProcessModeActive.value;
    const piiReviewResult = piiReviewFlow.scanSubmissionForPii({
      text: queryTextTrimmed,
      // Query language may be auto-detected server-side; the UI locale picks PII rules
      locale: state.selectedLanguage.value || state.uiLocale?.value || 'en',
      useTextProcessMode,
      isAutoSubmit,
    });

    if (piiReviewResult.hasFindings) {
      clearAutoSubmitQueryParamIfNeeded(route, router, isAutoSubmit, logService);
      return;
    }

    await submitQueryText({
      currentQuery: queryTextTrimmed,
      useTextProcessMode,
      isAutoSubmit,
      piiScanResult: piiReviewResult.scanResult,
      redactionApplied: false,
    });
  }

  function cancelQuery() {
    activeRequest?.abort();
  }

  async function continueWithPiiRedaction() {
    return piiReviewFlow.continueWithPiiRedaction({
      submitQueryText,
    });
  }

  function redactPiiInInput() {
    return piiReviewFlow.redactPiiInInput({
      setQueryText(value) {
        state.queryText.value = value;
      },
    });
  }

  return {
    fetchAvailableModels,
    setFallbackModel,
    applyUrlParametersAndAutoSubmit,
    submitQuery,
    cancelQuery,
    continueWithPiiRedaction,
    redactPiiInInput,
  };
}
