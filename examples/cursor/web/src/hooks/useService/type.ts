import { SWRConfiguration } from 'swr';

export type IService<Body, Data> = (body: Body) => Promise<Data>;
export type {
  Key as IServiceKey,
} from 'swr';
export type { SWRInfiniteConfiguration as IInfiniteCServiceConfiguration } from 'swr/infinite';
export type { SWRMutationConfiguration as IServiceMutationConfiguration } from 'swr/mutation';

export interface IServiceConfiguration<Data = any> extends SWRConfiguration<Data> {
  /**
   * The key of the service
   */
  key?: string;
}
