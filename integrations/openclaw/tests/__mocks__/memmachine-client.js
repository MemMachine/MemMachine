// Runtime stub for @memmachine/client used in Jest tests.
const MockMemory = {
  search: async () => null,
  add: async () => ({}),
  list: async () => null,
  delete: async () => ({}),
};
const MockProject = { memory: () => MockMemory };
class MemMachineClient {
  constructor(_config) {}
  project() { return MockProject; }
}
export default MemMachineClient;
