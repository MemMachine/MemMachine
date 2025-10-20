export class RequestError extends Error {
  public code: number;
  public url?: string;

  constructor(params: { url?: string; code: number; message: string }) {
    const { url, code, message } = params;
    super(message);
    this.name = 'RequestError';
    this.code = code;
    this.url = url;
  }
}
