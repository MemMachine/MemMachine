export class NoPermissionError extends Error {
  public code: number;
  public url?: string;

  constructor(params: {
    redirect?: string;
    url?: string;
    code: number;
    message: string;
  }) {
    const { code, message, url } = params;
    super(message);
    this.name = 'NoPermissionError';
    this.code = code;
    this.url = url;
  }
}
