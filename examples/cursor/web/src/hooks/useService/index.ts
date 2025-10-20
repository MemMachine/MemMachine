import { ServiceProvider } from './ServiceProvider';
import { useImmutableService } from './useImmutableService';
import { useInfiniteService } from './useInfiniteService';
import { useMutationService } from './useMutationService';
import { useService } from './useService';

type IUseService = typeof useService & {
  ServiceProvider: typeof ServiceProvider;
  useMutationService: typeof useMutationService;
  useInfiniteService: typeof useInfiniteService;
  useImmutableService: typeof useImmutableService;
};

(useService as IUseService).ServiceProvider = ServiceProvider;
(useService as IUseService).useMutationService = useMutationService;
(useService as IUseService).useInfiniteService = useInfiniteService;
(useService as IUseService).useImmutableService = useImmutableService;

export default useService as IUseService;
