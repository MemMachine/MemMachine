declare module "openclaw/plugin-sdk" {
  export type OpenClawPluginApi = {
    pluginConfig?: Record<string, unknown>;
    registerMemoryPromptSection?: (...args: any[]) => void;
    logger: {
      info: (...args: any[]) => void;
      warn: (...args: any[]) => void;
      error?: (...args: any[]) => void;
    };
    [key: string]: any;
  };

  export type MemoryPromptSectionBuilder = (...args: any[]) => string[];
  export function jsonResult(...args: any[]): any;
  export function readNumberParam(...args: any[]): any;
  export function readStringParam(...args: any[]): any;
}
