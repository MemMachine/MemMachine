export class InvalidTokenError extends Error {
  public code: number;
  public url?: string;
  public redirect?: string;
  public redirectUnauthorized?: boolean;

  constructor(params: {
    redirect?: string;
    redirectUnauthorized?: boolean;
    url?: string;
    code: number;
    message: string;
  }) {
    const { code, message, url, redirect, redirectUnauthorized } = params;
    super(message);
    this.name = 'InvalidTokenError';
    this.redirect = redirect;
    this.redirectUnauthorized = redirectUnauthorized;
    this.code = code;
    this.url = url;
  }
}
