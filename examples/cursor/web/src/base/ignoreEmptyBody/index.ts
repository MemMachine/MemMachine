/**
 * Ignore empty body params
 * @example ignoreEmptyBody({arr: [], b: ''}) => {}
 */
export default function ignoreEmptyBody(
  body: any,
  params?: {
    ignoreEmptyArray: boolean;
    ignoreEmptyStr: boolean;
  }
): any {
  const { ignoreEmptyArray = true, ignoreEmptyStr = true } = params ?? {};

  if (body === null || body === undefined) {
    return;
  }

  const isIgnoreEmptyStr = (v: any) =>
    typeof v === 'string' && v === '' && ignoreEmptyStr;
  const isIgnoreEmptyArray = (v: any) =>
    Array.isArray(v) && !v.length && ignoreEmptyArray;

  if (isIgnoreEmptyArray(body)) {
    return;
  }
  if (isIgnoreEmptyStr(body)) {
    return;
  }
  if (Array.isArray(body)) {
    return body;
  }
  if (typeof body === 'object') {
    return Object.fromEntries(
      Object.entries(body)
        .map(([k, v]) => {
          if (isIgnoreEmptyArray(v) || isIgnoreEmptyStr(v)) {
            return [];
          }
          if (v === null) {
            return [];
          }
          return [k, v];
        })
        .filter((v) => v.length)
    );
  }

  return body;
}
