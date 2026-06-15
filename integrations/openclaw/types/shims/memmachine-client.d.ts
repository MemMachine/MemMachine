declare module "@memmachine/client" {
  export type AddMemoryResult = any;
  export type EpisodicMemory = any;
  export type MemoryType = any;
  export type SearchMemoriesResult = any;
  export type SemanticMemory = any;

  export function __getMemMachineClientConfigs(): any[];
  export function __resetMemMachineClientConfigs(): void;

  export default class MemMachineClient {
    constructor(config: any);
    project(...args: any[]): any;
  }
}
