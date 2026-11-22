"""Local-only one-time promotion of an existing account to ADMIN.

Usage: python -m backend.promote_admin
Requires access to the same local PostgreSQL as FirstDay AI.
"""
from __future__ import annotations

import psycopg2

from backend.auth_routes import DB_CONFIG


def main() -> None:
    print('FirstDay AI — local administrator setup')
    email = input('Email of your existing account: ').strip().lower()
    if not email or '@' not in email:
        print('Invalid email. No changes made.')
        return
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id, full_name, role FROM employees WHERE email=%s', (email,))
            row = cur.fetchone()
            if not row:
                print('No account with this email. Register in the app first.')
                return
            print(f'Account: {row[1]} | current role: {row[2]}')
            answer = input('Type PROMOTE to grant ADMIN access: ').strip()
            if answer != 'PROMOTE':
                print('Cancelled. No changes made.')
                return
            cur.execute("UPDATE employees SET role='ADMIN' WHERE id=%s", (str(row[0]),))
    print('Administrator role granted. Sign out and sign in again.')


if __name__ == '__main__':
    main()
