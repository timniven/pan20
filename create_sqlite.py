"""Script to build the sqlite DB given the sqlalchemy schema."""
import os

from pan20.util.sqldb import orm, sqlite


db_name = 'pan20.db'


if __name__ == '__main__':
    # create data dir if not existing
    if not os.path.exists(sqlite.folder):
        os.mkdir(sqlite.folder)

    # path to the db file
    db_path = os.path.join(sqlite.folder, db_name)

    # remove old db files
    if os.path.exists(db_path):
        os.remove(db_path)

    # create the database
    engine = sqlite.create_engine(db_name)
    orm.Base.metadata.create_all(engine)
