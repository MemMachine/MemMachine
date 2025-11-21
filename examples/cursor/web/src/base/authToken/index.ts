import Cookies from 'js-cookie';

import * as storageKey from '@/constants/storageKey';

function set(token: string) {
  Cookies.set(storageKey.AUTH, token, {
    expires: 30,
  });
}

function remove() {
  Cookies.remove(storageKey.AUTH, {});
}

function get() {
  return Cookies.get(storageKey.AUTH);
}

export default {
  remove,
  set,
  get,
};
