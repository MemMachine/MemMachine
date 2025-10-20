import { TFunction } from 'i18next';
import { useCallback, useEffect } from 'react';

import error from '@/base/error';

const { RequestError, NetworkError, InvalidTokenError } = error;

interface IProps {
  toast: {
    error: (msg: string) => void;
  };
  onError?: (ev: PromiseRejectionEvent) => void;
  onAuthError?: (ev: PromiseRejectionEvent) => void;
  t: TFunction<'transaction', any>;
  loginPath?: string;
}

export default function useCatchError(props: IProps) {
  const { toast, t, onError, onAuthError } = props;

  const handle = useCallback(
    (ev: PromiseRejectionEvent) => {
      const error = ev.reason;
      console.error(error);
      if (error instanceof InvalidTokenError) {
        toast.error(error.message || t('requestError.default'));
        if (typeof onAuthError === 'function') {
          onAuthError(ev);
        }
        return;
      }
      if (error instanceof RequestError) {
        toast.error(error.message || t('requestError.default'));
      }
      if (error instanceof NetworkError) {
        toast.error(error.message || t('requestError.networkError'));
      }
      onError?.(ev);
    },
    [onError, toast, t, onAuthError]
  );

  useEffect(() => {
    window.addEventListener('unhandledrejection', handle);
    return () => window.removeEventListener('unhandledrejection', handle);
  }, [handle]);
}
