import ignoreEmptyBody from '../index';

describe('ignoreEmptyBody', () => {
  it('ignoreBody', () => {
    expect(
      ignoreEmptyBody('', {
        ignoreEmptyArray: false,
        ignoreEmptyStr: false,
      })
    ).toEqual('');

    expect(
      ignoreEmptyBody([], {
        ignoreEmptyArray: false,
        ignoreEmptyStr: false,
      })
    ).toEqual([]);

    expect(
      ignoreEmptyBody(
        { a: {}, b: '', c: 'a', d: [] },
        {
          ignoreEmptyArray: false,
          ignoreEmptyStr: false,
        }
      )
    ).toEqual({ a: {}, b: '', c: 'a', d: [] });

    expect(
      ignoreEmptyBody('', {
        ignoreEmptyArray: false,
        ignoreEmptyStr: true,
      })
    ).toBeUndefined();

    expect(
      ignoreEmptyBody(null, {
        ignoreEmptyArray: false,
        ignoreEmptyStr: true,
      })
    ).toBeUndefined();

    expect(
      ignoreEmptyBody(null, {
        ignoreEmptyArray: false,
        ignoreEmptyStr: true,
      })
    ).toBeUndefined();

    expect(
      ignoreEmptyBody('', {
        ignoreEmptyArray: true,
        ignoreEmptyStr: false,
      })
    ).toEqual('');

    expect(
      ignoreEmptyBody([], {
        ignoreEmptyArray: true,
        ignoreEmptyStr: false,
      })
    ).toBeUndefined();

    expect(
      ignoreEmptyBody(
        { a: {}, b: '', c: 'a', d: [] },
        {
          ignoreEmptyArray: true,
          ignoreEmptyStr: false,
        }
      )
    ).toEqual({ a: {}, b: '', c: 'a' });

    expect(
      ignoreEmptyBody(
        { a: {}, b: '', c: 'a', d: [] },
        {
          ignoreEmptyArray: false,
          ignoreEmptyStr: true,
        }
      )
    ).toEqual({ a: {}, c: 'a', d: [] });

    expect(
      ignoreEmptyBody(
        { a: {}, b: '', c: 'a', d: [] },
        {
          ignoreEmptyArray: true,
          ignoreEmptyStr: true,
        }
      )
    ).toEqual({ a: {}, c: 'a' });

    expect(
      ignoreEmptyBody(
        { a: '', d: [] },
        {
          ignoreEmptyArray: true,
          ignoreEmptyStr: true,
        }
      )
    ).toEqual({});

    expect(
      ignoreEmptyBody(
        { a: null },
        {
          ignoreEmptyArray: false,
          ignoreEmptyStr: true,
        }
      )
    ).toEqual({});

    expect(ignoreEmptyBody({ a: '', d: [] })).toEqual({});
  });
});
