function getContextOrThrow(getContext) {
  const context = getContext?.();

  if (
    !context ||
    typeof context.getState !== 'function' ||
    typeof context.setState !== 'function'
  ) {
    throw new Error('QueryInterface controller requires an explicit component context.');
  }

  return context;
}

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
  latestState,
  piiScanResult = null,
  redactionApplied = false,
}) {
  return {
    mode,
    textLength: String(text ?? '').length,
    selectedLanguage: latestState.selectedLanguage,
    extractionBackend:
      mode === 'textProcess' ? latestState.textProcessOptions?.extractionBackend || null : null,
    piiSummary: piiScanResult?.summary || { high: {}, review: {} },
    piiRedactionApplied: redactionApplied,
  };
}

function clearAutoSubmitQueryParamIfNeeded(context, isAutoSubmit, logService) {
  if (!isAutoSubmit && hasOwnQueryParam(context.routeQuery, 'autoSubmit')) {
    const newRouteQuery = { ...context.routeQuery };
    delete newRouteQuery.autoSubmit;

    Promise.resolve(context.replaceRouteQuery(newRouteQuery)).catch((error) => {
      if (error.name !== 'NavigationDuplicated' && error.name !== 'NavigationCancelled') {
        logService.warn('Error clearing autoSubmit from URL:', error);
      }
    });
  }
}

export function useQueryInterfaceController({ getContext, service, logService }) {
  let hasAppliedInitialUrlParameters = false;

  function setFallbackModel() {
    const context = getContextOrThrow(getContext);

    context.setState({
      availableModels: [{ text: 'BioLORD 2023-M', value: 'FremyCompany/BioLORD-2023-M' }],
      selectedModel: 'FremyCompany/BioLORD-2023-M',
    });

    logService.info('Using fallback embedding model', {
      model: context.getState().selectedModel,
    });
  }

  function applyUrlParametersAndAutoSubmit() {
    const context = getContextOrThrow(getContext);
    const queryParams = context.routeQuery ?? {};

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
      context.setState({ queryText: textParam });
    }

    if (modelParam !== undefined) {
      const validModels = context.getState().availableModels.map((model) => model.value);
      if (validModels.includes(modelParam)) {
        context.setState({ selectedModel: modelParam });
        advancedOptionsWereSet = true;
      }
    }

    if (thresholdParam !== undefined) {
      const parsedThreshold = parseFloat(thresholdParam);
      if (!Number.isNaN(parsedThreshold) && parsedThreshold >= 0 && parsedThreshold <= 1) {
        context.setState({ similarityThreshold: parsedThreshold });
        advancedOptionsWereSet = true;
      }
    }

    if (forceEndpointModeParam !== undefined) {
      context.setState({ forceEndpointMode: forceEndpointModeParam });
      advancedOptionsWereSet = true;
    }

    if (chunkingStrategyParam !== undefined) {
      context.setState({ chunkingStrategy: chunkingStrategyParam });
      advancedOptionsWereSet = true;
    }

    if (advancedOptionsWereSet) {
      context.setState({ showAdvancedOptions: true });
    }

    const currentState = context.getState();
    const performAutoSubmit =
      autoSubmitParam !== undefined ? parseBooleanParam(autoSubmitParam) : textParam !== undefined;

    if (
      performAutoSubmit &&
      typeof currentState.queryText === 'string' &&
      currentState.queryText.trim()
    ) {
      logService.info('Auto-submitting query based on URL parameters.');
      context.nextTick(() => setTimeout(() => submitQuery(true), 300));
    }
  }

  async function fetchAvailableModels() {
    const context = getContextOrThrow(getContext);

    context.setState({ modelsLoading: true });

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

        context.setState({
          availableModels,
          selectedModel,
        });

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
      context.setState({ modelsLoading: false });

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
    const context = getContextOrThrow(getContext);

    context.setState({
      isLoading: true,
      shouldScrollToTop: true,
      userHasScrolled: false,
    });

    const queryId = context.conversationStore.addQuery({
      query: rawQueryForHistory,
      redactedQuery: redactedQueryForHistory,
      loading: true,
      type: useTextProcessMode ? 'textProcess' : 'query',
    });

    if (useTextProcessMode) {
      context.setExpandedUserNote(queryId, true);
    }

    if (!isAutoSubmit) {
      context.setState({ queryText: '' });
    }

    try {
      let response;
      const latestState = context.getState();
      const sanitizedLog = getSanitizedSubmissionLog({
        mode: useTextProcessMode ? 'textProcess' : 'query',
        text: currentQuery,
        latestState,
        piiScanResult,
        redactionApplied,
      });

      if (useTextProcessMode) {
        const textProcessPayload = {
          text: currentQuery,
          extractionBackend: latestState.textProcessOptions.extractionBackend,
          llmModel: latestState.textProcessOptions.llmModel,
          llmMode: latestState.textProcessOptions.llmMode,
          language: latestState.selectedLanguage,
          chunkingStrategy: latestState.chunkingStrategy,
          windowSize: latestState.windowSize,
          stepSize: latestState.stepSize,
          splitThreshold: latestState.splitThreshold,
          minSegmentLength: latestState.minSegmentLength,
          semanticModelForChunking:
            latestState.semanticModelForChunking || latestState.selectedModel,
          retrievalModelForTextProcess:
            latestState.retrievalModelForTextProcess || latestState.selectedModel,
          trustRemoteCode: true,
          chunkRetrievalThreshold: latestState.chunkRetrievalThreshold,
          numResultsPerChunk: latestState.numResultsPerChunk,
          noAssertionDetectionForTextProcess: latestState.noAssertionDetectionForTextProcess,
          assertionPreferenceForTextProcess: latestState.assertionPreferenceForTextProcess,
          aggregatedTermConfidence: latestState.aggregatedTermConfidence,
          topTermPerChunkForAggregation: latestState.topTermPerChunkForAggregation,
          includeDetails: latestState.includeDetails,
        };
        logService.info('Submitting text request', sanitizedLog);
        response = await service.processText(textProcessPayload);
      } else {
        const queryPayload = {
          text: currentQuery,
          model_name: latestState.selectedModel,
          language: latestState.selectedLanguage,
          num_results: latestState.numResults,
          similarity_threshold: latestState.similarityThreshold,
          query_assertion_language: latestState.selectedLanguage,
          detect_query_assertion: true,
          include_details: latestState.includeDetails,
        };
        logService.info('Submitting text request', sanitizedLog);
        response = await service.queryHpo(queryPayload);
      }

      context.conversationStore.updateQueryResponse(queryId, response);

      if (useTextProcessMode) {
        context.fullTextWorkspaceStore.initializeTurn(queryId);
        context.fullTextWorkspaceStore.setExpanded(queryId, true);
      }
    } catch (error) {
      context.conversationStore.updateQueryResponse(queryId, null, error);
      logService.error('Error submitting query/processing text', error);
    } finally {
      context.setState({ isLoading: false });
      clearAutoSubmitQueryParamIfNeeded(context, isAutoSubmit, logService);
    }
  }

  async function submitQuery(isAutoSubmit = false) {
    const context = getContextOrThrow(getContext);
    const state = context.getState();
    const queryTextTrimmed = typeof state.queryText === 'string' ? state.queryText.trim() : '';

    if (!queryTextTrimmed) {
      logService.warn('Empty query submission prevented');
      return;
    }

    const useTextProcessMode = state.isTextProcessModeActive;
    const piiReviewResult = context.piiReviewFlow.scanSubmissionForPii({
      text: queryTextTrimmed,
      locale: state.selectedLanguage || 'en',
      useTextProcessMode,
      isAutoSubmit,
    });

    if (piiReviewResult.hasFindings) {
      clearAutoSubmitQueryParamIfNeeded(context, isAutoSubmit, logService);
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

  async function continueWithPiiRedaction() {
    const context = getContextOrThrow(getContext);
    return context.piiReviewFlow.continueWithPiiRedaction({
      submitQueryText,
    });
  }

  function redactPiiInInput() {
    const context = getContextOrThrow(getContext);
    return context.piiReviewFlow.redactPiiInInput({
      setQueryText(value) {
        context.setState({ queryText: value });
      },
    });
  }

  return {
    fetchAvailableModels,
    setFallbackModel,
    applyUrlParametersAndAutoSubmit,
    submitQuery,
    continueWithPiiRedaction,
    redactPiiInInput,
  };
}
