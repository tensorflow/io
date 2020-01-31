show server_version;
drop database test_db;
create database test_db;
\c test_db;
drop table test_table;
create table test_table(id bigint PRIMARY KEY, i32 int, i64 bigint, f32 float(4), f64 double precision);
insert into test_table(id, i32, i64, f32, f64) select i, i+1000, i+2000, i+3000, i+4000 from generate_series(0, 9) s(i);
select id, i32, i64, f32, f64 from test_table;
