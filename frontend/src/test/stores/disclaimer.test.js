import { describe, it, expect, beforeEach } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';

import { useDisclaimerStore } from '../../stores/disclaimer';

describe('disclaimer store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  it('tracks the disclaimer version acknowledged by the user', () => {
    const store = useDisclaimerStore();

    expect(store.isCurrentVersionAcknowledged).toBe(false);

    store.saveAcknowledgment();

    expect(store.isAcknowledged).toBe(true);
    expect(store.acknowledgedVersion).toBe(store.DISCLAIMER_VERSION);
    expect(store.isCurrentVersionAcknowledged).toBe(true);
  });
});
