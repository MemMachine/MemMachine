import { useCallback, useEffect, useRef } from 'react';
import { Middleware } from 'swr';

// A SWR middleware，is used to save data when key change.
export const laggy: Middleware = (useSWRNext) => {
  return (key, fetcher, config) => {
    const laggyDataRef = useRef<any>(undefined);

    const swr = useSWRNext(key, fetcher, config);

    useEffect(() => {
      if (swr.data !== undefined) {
        laggyDataRef.current = swr.data;
      }
    }, [swr.data]);

    const resetLaggy = useCallback(() => {
      laggyDataRef.current = undefined;
    }, []);

    const dataOrLaggyData =
      swr.data === undefined ? laggyDataRef.current : swr.data;

    const isLagging =
      swr.data === undefined && laggyDataRef.current !== undefined;

    return Object.assign({}, swr, {
      data: dataOrLaggyData,
      isLagging,
      resetLaggy,
    });
  };
};
