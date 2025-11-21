/* eslint-disable @typescript-eslint/no-explicit-any */
export type IRequestResult<T> = T & {
  errCode?: number;
  errMsg?: string;
};

export type IRequestParams<IData> = Omit<RequestInit, 'body'> & {
  url: string;
  body?: any;
  query?: any;
  baseURL?: string;
  timeout?: number;
  file?: boolean;
  text?: boolean;
  rawResponse?: boolean;
  ignoreEmptyStr?: boolean;
  ignoreEmptyArray?: boolean;
  redirectUnauthorized?: boolean;
  transResult?: (result: IData, response: Response) => IRequestResult<IData>;
  getAuthToken?: () => string | null | undefined;
  getCSRFToken?: () => string | null | undefined;
  getLanguage?: () => string | null | undefined;
};

export interface ICreateRequestParams {
  getAuthToken?: () => string | null | undefined;
  getCSRFToken?: () => string | null | undefined;
  getLanguage?: () => string | null | undefined;
  lng?: string;
  baseURL?: string;
  version?: string;
  ignoreEmptyStr?: boolean;
  ignoreEmptyArray?: boolean;
}

export type IRequest = <IData>(params: IRequestParams<IData>) => Promise<IData>;
