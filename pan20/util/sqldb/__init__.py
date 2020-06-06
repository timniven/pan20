from . import hsdbi, orm, sqlite


class DBInterface(hsdbi.SQLFacade):

    def __init__(self):
        super().__init__(connection_string=sqlite.connection_string('pan20.db'))
        self.docs = DocRepository(self.session)
        self.authors = AuthorRepository(self.session)
        self.fandoms = FandomRepository(self.session)
        self.ngrams = NGramRepository(self.session)


class DocRepository(hsdbi.SQLRepository):

    def __init__(self, session):
        super().__init__(
            primary_keys=['id'],
            class_type=orm.Doc,
            orm_module='pan20.util.sqldb.orm',
            session=session)


class AuthorRepository(hsdbi.SQLRepository):

    def __init__(self, session):
        super().__init__(
            primary_keys=['id'],
            class_type=orm.Author,
            orm_module='pan20.util.sqldb.orm',
            session=session)


class FandomRepository(hsdbi.SQLRepository):

    def __init__(self, session):
        super().__init__(
            primary_keys=['name'],
            class_type=orm.Fandom,
            orm_module='pan20.util.sqldb.orm',
            session=session)


class NGramRepository(hsdbi.SQLRepository):

    def __init__(self, session):
        super().__init__(
            primary_keys=['gram'],
            class_type=orm.NGram,
            orm_module='pan20.util.sqldb.orm',
            session=session)
