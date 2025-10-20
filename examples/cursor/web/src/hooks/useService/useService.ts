import { useCallback } from 'react';
import useSwr from 'swr';

import { IService, IServiceConfiguration, IServiceKey } from './type';

function useService<Body, Data>(
  service: IService<Body, Data> | null,
  body?: Body | null,
  options?: IServiceConfiguration<Data>
) {
  const fetcher = useCallback(() => {
    return service!(body!)
  }, [service, body]);

  let key: IServiceKey = null;
  if (service) {
    if (options?.key) {
      key =  options.key;
    } else {
      key = [service, body]
    }
  }

  const result = useSwr<Data>(
    key,
    fetcher,
    options
  );

  return result;
}

export { useService };
