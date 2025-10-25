import { PropsWithChildren } from 'react';
import { SWRConfig } from 'swr';

import { laggy } from './laggy';
import { IServiceConfiguration } from './type';

const defaultConfig: IServiceConfiguration = {
  errorRetryCount: 2,
};

export function ServiceProvider(
  props: PropsWithChildren<IServiceConfiguration>
) {
  const { children, use, ...restProps } = props;

  const onError = (e: Error) => {
    throw e;
  };
  return (
    <SWRConfig
      value={{
        ...defaultConfig,
        ...restProps,
        onError,
        use: [...(use ?? []), laggy],
      }}
    >
      {children}
    </SWRConfig>
  );
}
