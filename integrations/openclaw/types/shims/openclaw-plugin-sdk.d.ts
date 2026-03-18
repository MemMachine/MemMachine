declare module "openclaw/plugin-sdk" {
  export type OpenClawPluginApi = {
    pluginConfig?: Record<string, unknown>;
    logger: {
      info: (...args: any[]) => void;
      warn: (...args: any[]) => void;
      error?: (...args: any[]) => void;
    };
    [key: string]: any;
  };

  export function jsonResult(...args: any[]): any;
  export function readNumberParam(...args: any[]): any;
  export function readStringParam(...args: any[]): any;
}
