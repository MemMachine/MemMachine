// Runtime stub for @memmachine/client used in Jest tests.
const state = {
  listResults: [],
  searchResults: [],
};

export const mockMemory = {
  listCalls: [],
  searchCalls: [],
  addCalls: [],
  deleteCalls: [],
  async search(query, options) {
    this.searchCalls.push({ query, options });
    const result = state.searchResults.shift();
    if (result instanceof Error) throw result;
    return result ?? null;
  },
  async add(content, options) {
    this.addCalls.push({ content, options });
    return {};
  },
  async list(options) {
    this.listCalls.push(options);
    const result = state.listResults.shift();
    if (result instanceof Error) throw result;
    return result ?? null;
  },
  async delete(id, type) {
    this.deleteCalls.push({ id, type });
    return {};
  },
};

export function resetMockMemMachine() {
  state.listResults = [];
  state.searchResults = [];
  mockMemory.listCalls = [];
  mockMemory.searchCalls = [];
  mockMemory.addCalls = [];
  mockMemory.deleteCalls = [];
}

export function queueListResults(...results) {
  state.listResults.push(...results);
}

export function queueSearchResults(...results) {
  state.searchResults.push(...results);
}

const MockProject = {
  memory: () => mockMemory,
  getEpisodicCount: async () => 0,
};

class MemMachineClient {
  constructor(_config) {}
  project() { return MockProject; }
}
export default MemMachineClient;
