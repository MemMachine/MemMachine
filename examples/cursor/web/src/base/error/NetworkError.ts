export class NetworkError extends Error {
  public status: number;
  public statusText: string;
  public url: string;

  constructor(params: { url: string; status: number; statusText: string }) {
    const { url, status, statusText } = params;
    super(statusText);
    this.name = 'NetworkError';
    this.url = url;
    this.status = status;
    this.statusText = statusText;
  }
}
