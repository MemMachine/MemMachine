import Cookies from 'js-cookie';

import createRequest from '@/base/createRequest';
import * as storageKey from '@/constants/storageKey';

export const request = createRequest({
  getCSRFToken() {
    return Cookies.get(storageKey.CSRF)
  },
  getLanguage() {
    return localStorage.getItem('i18nextLng') ?? 'en-US';
  },
  version: '1.0.0',
});
