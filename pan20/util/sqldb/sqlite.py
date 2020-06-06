"""For connecting to sqlite."""
import sqlalchemy as db
from sqlalchemy import orm


# relative to the project root, where all code is executed
folder = 'data/sqlite'


def connection_string(db_name):
    return f'sqlite:///{folder}/{db_name}.db'


def create_engine(db_name):
    return db.create_engine(connection_string(db_name), echo=True)


def create_session(db_name):
    engine = create_engine(db_name)
    return orm.sessionmaker(bind=engine)()
