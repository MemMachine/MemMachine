declare module "@memmachine/client" {
  export type MemoryType = "episodic" | "semantic";

  export type AddMemoryResult = {
    results: Array<{ uid: string }>;
  };

  export type EpisodicMemory = {
    uid: string;
    score?: number | null;
    content: string;
    created_at?: string | null;
    producer_id: string;
    producer_role: string;
    produced_for_id?: string | null;
    episode_type?: string | null;
    metadata?: Record<string, unknown> | null;
  };

  export type ListEpisodicMemory = {
    uid: string;
    content: string;
    session_key: string;
    created_at: string;
    producer_id: string;
    producer_role: string;
    produced_for_id?: string | null;
    sequence_num?: number;
    episode_type?: string;
    content_type?: string;
    filterable_metadata?: Record<string, unknown> | null;
    metadata?: Record<string, unknown> | null;
  };

  export type SemanticMemory = {
    set_id: string;
    category: string;
    tag: string;
    feature_name: string;
    value: string;
    metadata: {
      citations?: string[];
      id?: string;
      other?: Record<string, unknown>;
    };
  };

  export type SearchMemoriesResult = {
    status: number;
    content: {
      episodic_memory?: {
        long_term_memory?: { episodes: EpisodicMemory[] };
        short_term_memory?: { episodes: EpisodicMemory[]; episode_summary?: string[] };
      } | null;
      semantic_memory?: SemanticMemory[] | null;
    };
  };

  export type ListMemoriesResult = {
    status: number;
    content: {
      episodic_memory?: ListEpisodicMemory[] | null;
      semantic_memory?: SemanticMemory[] | null;
    };
  };

  type Memory = {
    search(query: string, options?: Record<string, unknown>): Promise<SearchMemoriesResult>;
    add(content: string, options?: Record<string, unknown>): Promise<AddMemoryResult>;
    list(options?: Record<string, unknown>): Promise<ListMemoriesResult>;
    delete(ids: string | string[], type: MemoryType): Promise<void>;
  };

  type Project = {
    memory(context?: Record<string, string | undefined>): Memory;
    getEpisodicCount(): Promise<number>;
  };

  export default class MemMachineClient {
    constructor(config: { api_key?: string; base_url?: string });
    project(context: { org_id: string; project_id: string }): Project;
  }
}
