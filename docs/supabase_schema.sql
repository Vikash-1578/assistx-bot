-- Run once in Supabase: SQL Editor -> New query -> paste -> Run.
create table if not exists assistx_messages (
  id         bigint generated always as identity primary key,
  user_id    bigint not null,
  role       text   not null,
  content    text   not null,
  created_at timestamptz not null default now()
);
create index if not exists idx_assistx_messages_user on assistx_messages(user_id, id);

create table if not exists assistx_profiles (
  user_id bigint primary key,
  summary text  not null default '',
  facts   jsonb not null default '[]'::jsonb
);

-- The bot uses the service_role key (server side only). Block public access:
alter table assistx_messages enable row level security;
alter table assistx_profiles enable row level security;
