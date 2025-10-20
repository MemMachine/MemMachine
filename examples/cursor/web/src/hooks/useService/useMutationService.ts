import { useCallback } from 'react';
import useSWRMutation from 'swr/mutation';

import { IService, IServiceMutationConfiguration } from './type';

function useMutationService<Body, Data>(
  service: IService<Body, Data>,
  options?: IServiceMutationConfiguration<Data, any, any, any>
) {
  const fetcher = useCallback(
    ([func]: [IService<Body, Data>], opt: { arg: Body }) => {
      return func(opt.arg);
    },
    []
  );

  const result = useSWRMutation([service], fetcher, options);
  return result;
}

export { useMutationService };
