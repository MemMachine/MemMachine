import { useCallback } from 'react';
import useSWRImmutable from 'swr/immutable';

import { IService, IServiceConfiguration, IServiceKey } from './type';

function useImmutableService<Body, Data>(
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
  const result = useSWRImmutable<Data>(
    key,
    fetcher,
    options
  );

  return result;
}

export { useImmutableService };
