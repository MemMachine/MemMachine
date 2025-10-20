import { useCallback } from 'react';
import useSWRInfinite from 'swr/infinite';

import { IInfiniteCServiceConfiguration, IService } from './type';

function useInfiniteService<Body, Data>(
  service: IService<Body, Data>,
  body: (pageIndex: number, preData: Data) => Body | null,
  options?: IInfiniteCServiceConfiguration<Data>
) {
  const fetcher = useCallback(
    (params: Body) => {
      return service(params);
    },
    [service]
  );

  const result = useSWRInfinite<Data>(body as any, fetcher, {
    revalidateFirstPage: false,
    revalidateAll: false,
    ...options,
  });
  return result;
}

export { useInfiniteService };
