import errorCode from '../../constants/errorCode';
import authToken from '../authToken';
import error from '../error';
import ignoreEmptyBody from '../ignoreEmptyBody';
import { ICreateRequestParams, IRequestParams } from './type';

const { RequestError, NetworkError, InvalidTokenError, NoPermissionError } =
  error;

const lngMap: Record<string, string> = {
  'en-US': 'en',
  'zh-CN': 'zh',
  'zh-TW': 'zh-TW',
};

export const InvalidTokenStatus = [
  errorCode.UNAUTHORIZED,
];

export const NoPermissionStatus = [errorCode.FORBIDDEN];

export function createRequest(topParams: ICreateRequestParams) {

  async function request<IData>(params: IRequestParams<IData>) {
    let {
      url,
    } = params;
    const {
      method = 'GET',
      baseURL = '',
      body,
      query,
      headers,
      timeout = 30000,
      file = false,
      text = false,
      redirectUnauthorized = true,
      transResult,
      ...restParams
    } = params;
    // const getAuthToken = params.getAuthToken ?? topParams.getAuthToken;
    const getCSRFToken = params.getCSRFToken ?? topParams.getCSRFToken;
    const getLanguage = params.getLanguage ?? topParams.getLanguage;
    if (import.meta.env.VITE_RANCHER_PROXY_TARGET) {
      url = url.replace(import.meta.env.VITE_RANCHER_PROXY_TARGET, '');
    }

    const ignoreEmptyArray =
      params.ignoreEmptyArray ?? topParams.ignoreEmptyArray ?? true;
    const ignoreEmptyStr =
      params.ignoreEmptyStr ?? topParams.ignoreEmptyStr ?? true;

    let language = 'en';
    if (getLanguage) {
      const languageKey = getLanguage();
      language = languageKey ? lngMap[languageKey] : 'en';
    }
    const requestInit: RequestInit = {
      method,
      headers: {
        Accept: 'application/json',
        Locale: language,
      },
      body,
      ...restParams,
    };

    if (!file) {
      (requestInit.headers as Record<string, string>)['Content-Type'] =
        'application/json';
    }

    if (headers) {
      (requestInit.headers as Record<string, any>) = {
        ...requestInit.headers,
        ...headers,
      };
    }
    if (getCSRFToken) {
      const csrf = getCSRFToken();
      if (csrf) {
        (requestInit.headers as Record<string, any>) = {
          ...requestInit.headers,
          'x-api-csrf': csrf,
        };
      }
    }

    const token = authToken.get();
    if (token !== null && token !== undefined) {
      requestInit.headers = {
        ...requestInit.headers,
        Authorization: `Bearer ${token}`,
      };
    }

    if (['post', 'put', 'patch', 'delete'].includes(method?.toLowerCase())) {
      if (file) {
        const formData = new FormData();
        Object.entries(body).forEach(([k, item]) => {
          if (Array.isArray(item)) {
            item.forEach((v) => {
              formData.append(k, v);
            });
          } else {
            formData.append(k, item as any);
          }
        });
        requestInit.body = formData;
      } else {
        requestInit.body = ignoreEmptyBody(body, {
          ignoreEmptyArray,
          ignoreEmptyStr,
        });

        requestInit.body = JSON.stringify(requestInit.body);
      }
    }

    const queryString = query ? `?${new URLSearchParams(query)}` : '';

    const controller = new AbortController();
    requestInit.signal = controller.signal;
    setTimeout(() => {
      controller.abort();
    }, timeout);

    const response = await fetch(baseURL + url + queryString, requestInit);
    let result: any = null;

    try {
      result = (await response.text());
    } catch (error) {
      console.error('Response parse to text failed\n', error);
    }

    if (!text && result !== '' && result !== null) {
      try {
        result = await JSON.parse(result);
        result.__warning = response.headers.get('warning') || '';
      } catch (error) {
        console.error('Response parse to json failed\n', error);
      }
    }

    if (!response.ok) {
      throw new NetworkError({
        status: response.status,
        statusText: result?.message ?? result ?? response.statusText,
        url,
      });
    }

    if (!result.success) {
      if (result && InvalidTokenStatus.includes(parseInt(result.status))) {
        throw new InvalidTokenError({
          code: result?.code ?? response.status,
          message: result?.message ?? response.statusText,
          redirectUnauthorized,
          url,
        });
      }

      if (result && NoPermissionStatus.includes(result.errorCode)) {
        throw new NoPermissionError({
          code: result?.code ?? response.status,
          message: result?.message ?? response.statusText,
          url,
        });
      }

      if (response.status >= 400 && response.status < 500) {
        throw new RequestError({
          code: result?.code ?? response.status,
          message: result?.message ?? result ?? response.statusText,
          url,
        });
      }

      throw new NetworkError({
        status: response.status,
        statusText: result?.message ?? result ?? response.statusText,
        url,
      });
    }

    // if (!result.success) {

    // }

    if (typeof transResult === 'function') {
      return transResult(result, response) as IData;
    }

    return result as IData;
  }
  return request;
}
