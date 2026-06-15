// Runtime stub for @memmachine/client used in Jest tests.
const MockMemory = {
  search: async () => null,
  add: async () => ({}),
  list: async () => null,
  delete: async () => ({}),
};
const MockProject = { memory: () => MockMemory };
const memMachineClientConfigs = [];
class MemMachineClient {
  constructor(config) {
    memMachineClientConfigs.push(config);
  }
  project() { return MockProject; }
}
export function __getMemMachineClientConfigs() {
  return [...memMachineClientConfigs];
}
export function __resetMemMachineClientConfigs() {
  memMachineClientConfigs.length = 0;
}
export default MemMachineClient;
