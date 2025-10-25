
export interface IRespBase<T> {
  code: number;
  data: T;
  message: string;
  success: boolean;
}
