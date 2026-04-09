import { ref } from 'vue';
import { useConversationStore } from '../stores/conversation';
import { useFileDownload } from './useFileDownload';
import { logService } from '../services/logService';

/**
 * Composable for managing phenotype collection state and actions.
 * Extracted from QueryInterface.vue to reduce component size.
 */
export function usePhenotypeCollection() {
  const conversationStore = useConversationStore();
  const { downloadText, downloadJson } = useFileDownload();

  // Subject information for Phenopacket export
  const phenopacketSubjectId = ref('');
  const phenopacketSex = ref(null);
  const phenopacketDateOfBirth = ref(null);

  function addPhenotype(phenotype, queryAssertionStatus = null) {
    logService.info('Adding phenotype to collection', {
      phenotype,
      queryAssertionStatus,
      phenotypeHasAssertionStatus: phenotype.assertion_status ? true : false,
      phenotypeAssertionStatus: phenotype.assertion_status,
    });

    const added = conversationStore.addPhenotype(phenotype, queryAssertionStatus);
    if (added) {
      logService.debug('Phenotype added to collection via store', {
        hpo_id: phenotype.hpo_id,
      });
    } else {
      logService.debug('Phenotype was duplicate, not added', {
        hpo_id: phenotype.hpo_id,
      });
    }
  }

  function removePhenotype(index) {
    const phenotype = conversationStore.collectedPhenotypes[index];
    logService.info('Removing phenotype from collection', {
      index,
      phenotype,
    });
    conversationStore.removePhenotype(index);
  }

  function toggleAssertionStatus(index) {
    conversationStore.toggleAssertionStatus(index);
    const phenotype = conversationStore.collectedPhenotypes[index];
    if (phenotype) {
      logService.info('Toggled phenotype assertion status', {
        hpo_id: phenotype.hpo_id,
        newStatus: phenotype.assertion_status,
      });
    }
  }

  function clearCollection() {
    logService.info('Clearing phenotype collection and subject information');
    conversationStore.clearPhenotypes();
    phenopacketSubjectId.value = '';
    phenopacketSex.value = null;
    phenopacketDateOfBirth.value = null;
  }

  function toggleCollectionPanel() {
    conversationStore.toggleCollectionPanel();
  }

  function exportCollectionAsText() {
    const phenotypes = conversationStore.collectedPhenotypes;
    logService.info('Exporting phenotypes as text', { count: phenotypes.length });
    let exportText = 'HPO Phenotypes Collection\n';
    exportText += 'Exported on: ' + new Date().toLocaleString() + '\n\n';
    exportText += 'ID\tLabel\tAssertion Status\n';
    phenotypes.forEach((p) => {
      exportText += `${p.hpo_id}\t${p.label}\t${p.assertion_status || 'affirmed'}\n`;
    });
    downloadText(exportText, 'hpo_phenotypes.txt');
  }

  function exportAsPhenopacket() {
    const phenotypes = conversationStore.collectedPhenotypes;
    if (phenotypes.length === 0) {
      logService.warn('Attempted to export empty phenopacket collection');
      return;
    }
    logService.info('Starting Phenopacket export process', {
      count: phenotypes.length,
    });
    try {
      const timestamp = new Date().toISOString();
      const phenopacketId = `phentrieve-export-${Date.now()}`;
      const phenopacket = {
        id: phenopacketId,
        metaData: {
          created: timestamp,
          createdBy: 'Phentrieve Frontend',
          phenopacketSchemaVersion: '2.0.0',
          resources: [
            {
              id: 'phentrieve',
              name: 'Phentrieve',
              namespacePrefix: 'Phentrieve',
              url: 'https://phentrieve.kidney-genetics.org/',
              version: import.meta.env.VITE_APP_VERSION || '1.0.0',
              iriPrefix: 'phentrieve',
            },
          ],
        },
        phenotypicFeatures: [],
      };
      if (
        phenopacketSubjectId.value ||
        phenopacketSex.value !== null ||
        phenopacketDateOfBirth.value
      ) {
        phenopacket.subject = {};
        if (phenopacketSubjectId.value) phenopacket.subject.id = phenopacketSubjectId.value.trim();
        if (phenopacketSex.value !== null) {
          const sexMap = { 0: 'UNKNOWN_SEX', 1: 'FEMALE', 2: 'MALE', 3: 'OTHER_SEX' };
          phenopacket.subject.sex = sexMap[phenopacketSex.value];
        }
        if (phenopacketDateOfBirth.value) {
          const dob = new Date(phenopacketDateOfBirth.value + 'T00:00:00Z');
          if (!isNaN(dob.getTime()))
            phenopacket.subject.timeAtLastEncounter = { timestamp: dob.toISOString() };
        }
        if (Object.keys(phenopacket.subject).length === 0) delete phenopacket.subject;
      }
      phenotypes.forEach((cp) => {
        phenopacket.phenotypicFeatures.push({
          type: { id: cp.hpo_id, label: cp.label },
          excluded: cp.assertion_status === 'negated',
        });
      });
      const filename = `phentrieve_phenopacket_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
      downloadJson(phenopacket, filename);
      logService.info('Phenopacket successfully exported', { filename });
    } catch (error) {
      logService.error('Error during Phenopacket export', { error });
      alert('Error exporting Phenopacket. Check console.');
    }
  }

  return {
    phenopacketSubjectId,
    phenopacketSex,
    phenopacketDateOfBirth,
    addPhenotype,
    removePhenotype,
    toggleAssertionStatus,
    clearCollection,
    toggleCollectionPanel,
    exportCollectionAsText,
    exportAsPhenopacket,
  };
}
